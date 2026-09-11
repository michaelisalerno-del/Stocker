import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from stocker_core.markets import ActivityScanner, CapBucket, MarketId, get_market
from stocker_core.runs import ACTIVITY_CAPACITY_V2_VERSION
from stocker_execution.activity_shortlist import (
    ActivityShortlistService,
    ActivityShortlistStatus,
    ActivityShortlistStore,
    ScannerCapabilities,
)
from stocker_execution.ibkr import IbkrError
from test_ibkr_resources import (
    CallbackEvent,
    LowLevelScannerClient,
    connected_stream_boundary,
    wait_until,
)


@pytest.mark.parametrize("warning_request_id", [71, 999])
def test_uk_discovery_retains_stock_results_and_audits_only_its_warnings(
    tmp_path, warning_request_id
):
    async def scenario():
        client = LowLevelScannerClient()
        client.errorEvent = CallbackEvent()
        broker = connected_stream_boundary(client)
        market = get_market(MarketId.UK_LSE)
        broker._scanner_capabilities = ScannerCapabilities(
            frozenset({market.scanner_location}),
            frozenset(component.value for component in ActivityScanner),
            frozenset(),
        )
        store = ActivityShortlistStore(tmp_path / "activity.sqlite")
        service = ActivityShortlistService(
            store,
            profile_id="ACTIVITY_CAPACITY_V2",
            profile_version=ACTIVITY_CAPACITY_V2_VERSION,
            watch_limit=5,
            allow_late_capture=True,
        )
        now = datetime(2026, 9, 9, 8, tzinfo=UTC)
        task = asyncio.create_task(
            service.get_or_create(
                broker,
                market=market,
                cap_bucket=CapBucket.ALL,
                session=now.date(),
                screen_at=now,
                now=now,
            )
        )
        contract = SimpleNamespace(
            symbol="TEST",
            conId=123,
            secType="STK",
            exchange="SMART",
            primaryExchange="LSE",
            currency="GBP",
        )
        for index in range(3):
            await wait_until(lambda: len(client.wrapper.futures), index + 1)
            client.errorEvent.emit(
                warning_request_id,
                492,
                "Additional permissions required for precise scanner results",
                None,
            )
            client.wrapper.futures[index].set_result(
                [SimpleNamespace(rank=0, contractDetails=SimpleNamespace(contract=contract))]
            )
        snapshot = await task
        assert snapshot.status is ActivityShortlistStatus.READY
        assert [(row.symbol, row.selected) for row in snapshot.candidates] == [("TEST", True)]
        assert bool(snapshot.reason) == (warning_request_id == 71)
        if warning_request_id == 71:
            assert "precision warning (492)" in snapshot.reason
        assert (
            store.get(
                market.market_id.value,
                CapBucket.ALL,
                now.date(),
                profile_id=snapshot.profile_id,
                profile_version=snapshot.profile_version,
            )
            == snapshot
        )
        assert client.cancelled_scanners == [71, 71, 71]
        assert len(client.errorEvent.handlers) == 1

    asyncio.run(scenario())


def test_unsupported_location_error_is_not_a_successful_empty_scan():
    async def scenario():
        client = LowLevelScannerClient()
        client.errorEvent = CallbackEvent()
        broker = connected_stream_boundary(client)
        market = get_market(MarketId.SOUTH_AFRICA_JSE)
        broker._scanner_capabilities = ScannerCapabilities(
            frozenset({market.scanner_location}),
            frozenset({"MOST_ACTIVE_AVG_USD"}),
            frozenset(),
        )
        task = asyncio.create_task(broker.activity_scan(
            market=market, cap_bucket=CapBucket.ALL,
            component=ActivityScanner.MOST_ACTIVE_AVG_USD, stock_type_filter="CORP",
        ))
        await wait_until(lambda: len(client.wrapper.futures), 1)
        client.errorEvent.emit(
            71, 162, "Market Scanner is not configured for one of the chosen locations.", None,
        )
        client.wrapper.futures[0].set_result([])
        with pytest.raises(IbkrError, match="not configured"):
            await task
        assert client.cancelled_scanners == [71]

    asyncio.run(scenario())
