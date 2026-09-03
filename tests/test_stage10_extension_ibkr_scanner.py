from __future__ import annotations

import asyncio
from types import SimpleNamespace

from stocker_core.config import IbkrConfig
from stocker_core.markets import ActivityScanner, CapBucket, MarketId, get_market
from stocker_core.runs import Environment
from stocker_execution.ibkr import IbkrConnection


class ScannerClient:
    def __init__(self) -> None:
        self.parameter_requests = 0
        self.scanner_requests: list[object] = []

    def isConnected(self) -> bool:
        return True

    async def reqScannerParametersAsync(self) -> str:
        self.parameter_requests += 1
        return """
        <ScanParameterResponse>
          <Location><locationCode>STK.US.MAJOR</locationCode></Location>
          <ScanType><scanCode>TOP_TRADE_RATE</scanCode></ScanType>
          <ScanType><scanCode>TOP_VOLUME_RATE</scanCode></ScanType>
          <ScanType><scanCode>HOT_BY_VOLUME</scanCode></ScanType>
          <RangeFilter><code>marketCapAbove</code></RangeFilter>
          <RangeFilter><code>marketCapBelow</code></RangeFilter>
        </ScanParameterResponse>
        """

    async def reqScannerDataAsync(
        self,
        subscription: object,
        scanner_subscription_options: list[object] | None = None,
        scanner_subscription_filter_options: list[object] | None = None,
    ) -> list[object]:
        assert scanner_subscription_options == []
        assert scanner_subscription_filter_options == []
        self.scanner_requests.append(subscription)
        contract = SimpleNamespace(
            symbol="MSFT",
            conId=272093,
            exchange="SMART",
            primaryExchange="NASDAQ",
            currency="USD",
            secType="STK",
        )
        return [
            SimpleNamespace(
                rank=0,
                contractDetails=SimpleNamespace(contract=contract),
            )
        ]


class CurrentAsxScannerClient(ScannerClient):
    def __init__(self) -> None:
        super().__init__()
        self.filter_options: list[object] = []

    async def reqScannerParametersAsync(self) -> str:
        self.parameter_requests += 1
        return """
        <ScanParameterResponse>
          <Location>
            <locationCode>STK.HK.ASX</locationCode>
            <instruments>STOCK.HK</instruments>
          </Location>
          <ScanType><scanCode>TOP_TRADE_RATE</scanCode></ScanType>
          <ScanType><scanCode>TOP_VOLUME_RATE</scanCode></ScanType>
          <ScanType><scanCode>HOT_BY_VOLUME</scanCode></ScanType>
          <RangeFilter><code>marketCapAbove1e6</code></RangeFilter>
          <RangeFilter><code>marketCapBelow1e6</code></RangeFilter>
        </ScanParameterResponse>
        """

    async def reqScannerDataAsync(
        self,
        subscription: object,
        scanner_subscription_options: list[object] | None = None,
        scanner_subscription_filter_options: list[object] | None = None,
    ) -> list[object]:
        assert scanner_subscription_options == []
        self.scanner_requests.append(subscription)
        self.filter_options = list(scanner_subscription_filter_options or [])
        contract = SimpleNamespace(
            symbol="BHP",
            conId=4036813,
            exchange="SMART",
            primaryExchange="ASX",
            currency="AUD",
            secType="STK",
        )
        return [SimpleNamespace(rank=0, contractDetails=SimpleNamespace(contract=contract))]


def connection(client: ScannerClient) -> IbkrConnection:
    broker = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=81,
            expected_account="DU123456",
        ),
        client=client,  # type: ignore[arg-type]
    )
    broker._account_id = "DU123456"
    return broker


def test_scanner_parameters_are_discovered_once_and_cached() -> None:
    client = ScannerClient()
    broker = connection(client)

    first = asyncio.run(broker.scanner_capabilities())
    second = asyncio.run(broker.scanner_capabilities())

    assert first is second
    assert client.parameter_requests == 1
    assert first.locations == frozenset({"STK.US.MAJOR"})
    assert first.scan_codes == frozenset({"TOP_TRADE_RATE", "TOP_VOLUME_RATE", "HOT_BY_VOLUME"})
    assert first.filters == frozenset({"marketCapAbove", "marketCapBelow"})


def test_scanner_capabilities_retain_location_specific_components() -> None:
    class LocationAwareClient(ScannerClient):
        async def reqScannerParametersAsync(self) -> str:
            self.parameter_requests += 1
            return """
            <ScanParameterResponse>
              <Location>
                <locationCode>STK.US.MAJOR</locationCode>
                <ScanType><scanCode>TOP_TRADE_RATE</scanCode></ScanType>
                <ScanType><scanCode>TOP_VOLUME_RATE</scanCode></ScanType>
                <RangeFilter><code>marketCapAbove</code></RangeFilter>
                <RangeFilter><code>marketCapBelow</code></RangeFilter>
              </Location>
              <Location>
                <locationCode>STK.EU</locationCode>
                <ScanType><scanCode>HOT_BY_VOLUME</scanCode></ScanType>
              </Location>
            </ScanParameterResponse>
            """

    capabilities = asyncio.run(connection(LocationAwareClient()).scanner_capabilities())
    assert capabilities.scan_codes_for("STK.US.MAJOR") == frozenset(
        {"TOP_TRADE_RATE", "TOP_VOLUME_RATE"}
    )
    assert capabilities.scan_codes_for("STK.EU") == frozenset({"HOT_BY_VOLUME"})
    assert capabilities.filters_for("STK.US.MAJOR") == frozenset(
        {"marketCapAbove", "marketCapBelow"}
    )


def test_activity_scan_sends_exact_market_cap_component_and_bound() -> None:
    client = ScannerClient()
    broker = connection(client)

    rows = asyncio.run(
        broker.activity_scan(
            market=get_market(MarketId.US_NASDAQ),
            cap_bucket=CapBucket.MID,
            component=ActivityScanner.TOP_TRADE_RATE,
            max_results=50,
        )
    )

    subscription = client.scanner_requests[0]
    assert subscription.locationCode == "STK.US.MAJOR"  # type: ignore[attr-defined]
    assert subscription.scanCode == "TOP_TRADE_RATE"  # type: ignore[attr-defined]
    assert subscription.numberOfRows == 50  # type: ignore[attr-defined]
    assert subscription.marketCapAbove == 2_000  # type: ignore[attr-defined]
    assert subscription.marketCapBelow == 10_000  # type: ignore[attr-defined]
    assert rows[0].symbol == "MSFT"
    assert rows[0].con_id == 272093
    assert rows[0].rank == 1


def test_asx_activity_scan_uses_live_ibkr_scanner_identity_and_cap_filter_names() -> None:
    client = CurrentAsxScannerClient()
    broker = connection(client)

    rows = asyncio.run(
        broker.activity_scan(
            market=get_market(MarketId.AUSTRALIA_ASX),
            cap_bucket=CapBucket.MID,
            component=ActivityScanner.TOP_TRADE_RATE,
            max_results=50,
        )
    )

    subscription = client.scanner_requests[0]
    assert subscription.instrument == "STOCK.HK"  # type: ignore[attr-defined]
    assert subscription.locationCode == "STK.HK.ASX"  # type: ignore[attr-defined]
    assert [(item.tag, item.value) for item in client.filter_options] == [  # type: ignore[attr-defined]
        ("marketCapAbove1e6", "2000"),
        ("marketCapBelow1e6", "10000"),
    ]
    assert rows[0].symbol == "BHP"
