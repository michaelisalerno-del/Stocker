from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_core.config import IbkrConfig
from stocker_core.markets import ActivityScanner, CapBucket, MarketId, get_market
from stocker_core.runs import Environment
from stocker_execution.activity_shortlist import (
    ActivityCandidate,
    ActivityShortlistSnapshot,
    ActivityShortlistStatus,
    ActivityShortlistStore,
    rank_activity_candidates,
)
from stocker_execution.ibkr import (
    IbkrConnection,
    IbkrError,
    QualifiedInstrument,
    QualifiedOption,
)
from stocker_execution.ibkr_resources import simulate_capacity


class ConfigClient:
    def __init__(self) -> None:
        self.client = SimpleNamespace(MaxRequests=45, RequestsInterval=1)

    def isConnected(self) -> bool:
        return False


class CallbackEvent:
    def __init__(self) -> None:
        self.handlers: list[object] = []

    def __iadd__(self, handler: object) -> CallbackEvent:
        self.handlers.append(handler)
        return self

    def __isub__(self, handler: object) -> CallbackEvent:
        self.handlers.remove(handler)
        return self

    def emit(self, *args: object) -> None:
        for handler in tuple(self.handlers):
            handler(*args)  # type: ignore[operator]


class ErrorClient(ConfigClient):
    def __init__(self) -> None:
        super().__init__()
        self.errorEvent = CallbackEvent()


class StreamClient(ConfigClient):
    def __init__(self) -> None:
        super().__init__()
        self.connected = True
        self.tickers: dict[int, SimpleNamespace] = {}
        self.requested: list[int] = []
        self.cancelled: list[int] = []
        self.generic_tick_lists: list[str] = []
        self.raise_for: set[int] = set()

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def reqMarketDataType(self, market_data_type: int) -> None:
        assert market_data_type in {1, 2, 3, 4}

    def reqMktData(
        self,
        contract: object,
        genericTickList: str = "",
        snapshot: bool = False,
        regulatorySnapshot: bool = False,
    ) -> object:
        del snapshot, regulatorySnapshot
        con_id = int(contract.conId)  # type: ignore[attr-defined]
        if con_id in self.raise_for:
            raise RuntimeError("market data request failed")
        self.requested.append(con_id)
        self.generic_tick_lists.append(genericTickList)
        return self.tickers.setdefault(con_id, incomplete_ticker())

    def cancelMktData(self, contract: object) -> bool:
        self.cancelled.append(int(contract.conId))  # type: ignore[attr-defined]
        return True


class ConcurrencyClient(StreamClient):
    def __init__(self) -> None:
        super().__init__()
        self.history_gate = asyncio.Event()
        self.scanner_gate = asyncio.Event()
        self.active_history = 0
        self.peak_history = 0
        self.active_scanners = 0
        self.peak_scanners = 0
        self.parameter_requests = 0

    async def reqHistoricalDataAsync(self, contract: object, **kwargs: object) -> list[object]:
        del contract, kwargs
        self.active_history += 1
        self.peak_history = max(self.peak_history, self.active_history)
        try:
            await self.history_gate.wait()
            return [
                SimpleNamespace(
                    date=datetime(2026, 9, 1, 13, 30, tzinfo=UTC),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1_000,
                )
            ]
        finally:
            self.active_history -= 1

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
        del subscription, scanner_subscription_options, scanner_subscription_filter_options
        self.active_scanners += 1
        self.peak_scanners = max(self.peak_scanners, self.active_scanners)
        try:
            await self.scanner_gate.wait()
            return []
        finally:
            self.active_scanners -= 1


class ScannerFutureWrapper:
    def __init__(self) -> None:
        self.futures: list[asyncio.Future[list[object]]] = []

    def startReq(self, _request_id: int, *, container: object) -> asyncio.Future[list[object]]:
        del container
        future: asyncio.Future[list[object]] = asyncio.get_running_loop().create_future()
        self.futures.append(future)
        return future


class LowLevelScannerClient(ConcurrencyClient):
    def __init__(self) -> None:
        super().__init__()
        self.wrapper = ScannerFutureWrapper()
        self.cancelled_scanners: list[int] = []

    def reqScannerSubscription(
        self,
        subscription: object,
        scanner_subscription_options: list[object],
        scanner_subscription_filter_options: list[object],
    ) -> object:
        del subscription, scanner_subscription_options, scanner_subscription_filter_options
        return SimpleNamespace(reqId=71)

    def cancelScannerSubscription(self, data_list: object) -> None:
        self.cancelled_scanners.append(int(data_list.reqId))  # type: ignore[attr-defined]


class WatchlistScannerClient(ConcurrencyClient):
    async def reqScannerDataAsync(
        self,
        subscription: object,
        scanner_subscription_options: list[object] | None = None,
        scanner_subscription_filter_options: list[object] | None = None,
    ) -> list[object]:
        del subscription, scanner_subscription_options, scanner_subscription_filter_options
        return [
            SimpleNamespace(
                rank=index,
                contractDetails=SimpleNamespace(
                    contract=SimpleNamespace(
                        symbol=f"STK{index}",
                        conId=index + 1,
                        exchange="SMART",
                        primaryExchange="NASDAQ",
                        currency="USD",
                        secType="STK",
                    )
                ),
            )
            for index in range(50)
        ]


class CacheClient(StreamClient):
    def __init__(self) -> None:
        super().__init__()
        self.qualification_requests = 0
        self.option_chain_requests = 0

    async def connectAsync(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.connected = True

    def managedAccounts(self) -> list[str]:
        return ["DU123456"]

    async def qualifyContractsAsync(
        self, *contracts: object, returnAll: bool = False
    ) -> list[object]:
        del returnAll
        self.qualification_requests += 1
        contract = contracts[0]
        contract.conId = 265598  # type: ignore[attr-defined]
        contract.exchange = "SMART"  # type: ignore[attr-defined]
        contract.primaryExchange = "NASDAQ"  # type: ignore[attr-defined]
        contract.currency = "USD"  # type: ignore[attr-defined]
        contract.secType = "STK"  # type: ignore[attr-defined]
        return [contract]

    async def reqSecDefOptParamsAsync(
        self,
        underlyingSymbol: str,
        futFopExchange: str,
        underlyingSecType: str,
        underlyingConId: int,
    ) -> list[object]:
        del underlyingSymbol, futFopExchange, underlyingSecType
        self.option_chain_requests += 1
        return [
            SimpleNamespace(
                exchange="SMART",
                underlyingConId=underlyingConId,
                tradingClass="AAPL",
                multiplier="100",
                expirations=["20260918"],
                strikes=[100.0, 105.0],
            )
        ]


def config(*, market_data_line_budget: int = 100) -> IbkrConfig:
    return IbkrConfig(
        environment=Environment.PAPER,
        host="127.0.0.1",
        port=4002,
        client_id=91,
        market_data_line_budget=market_data_line_budget,
    )


def test_stocker_market_data_budget_defaults_to_100_and_reuses_ib_async_throttle() -> None:
    client = ConfigClient()
    connection = IbkrConnection(config(), client=client)  # type: ignore[arg-type]

    status = connection.resource_status()

    assert connection.config.market_data_line_budget == 100
    assert client.client.MaxRequests == 45
    assert status.market_data_line_budget == 100
    assert status.market_data_budget_label == "Stocker API line budget"
    assert status.ibkr_account_line_limit is None
    assert status.ib_async_max_requests == 45
    assert status.ib_async_requests_interval == 1


def test_lower_stocker_budget_lowers_existing_ib_async_throttle() -> None:
    client = ConfigClient()

    connection = IbkrConnection(config(market_data_line_budget=20), client=client)  # type: ignore[arg-type]

    assert connection.config.market_data_line_budget == 20
    assert client.client.MaxRequests == 10
    assert connection.resource_status().ib_async_max_requests == 10


def test_reconfigure_rederives_throttle_without_exceeding_library_limit() -> None:
    client = ConfigClient()
    connection = IbkrConnection(config(market_data_line_budget=20), client=client)  # type: ignore[arg-type]

    connection.reconfigure(config(market_data_line_budget=80))
    assert client.client.MaxRequests == 40

    connection.reconfigure(config(market_data_line_budget=200))
    assert client.client.MaxRequests == 45


def test_raised_stocker_budget_does_not_raise_above_library_limit() -> None:
    client = ConfigClient()

    connection = IbkrConnection(config(market_data_line_budget=200), client=client)  # type: ignore[arg-type]

    assert connection.config.market_data_line_budget == 200
    assert client.client.MaxRequests == 45


def test_resource_errors_are_sanitized_counted_and_do_not_disconnect() -> None:
    client = ErrorClient()
    connection = IbkrConnection(config(), client=client)  # type: ignore[arg-type]
    contract = SimpleNamespace(conId=265598, symbol="AAPL", exchange="SMART")

    client.errorEvent.emit(
        7,
        100,
        "Max rate of messages exceeded for account DU123456",
        contract,
    )
    pacing = connection.resource_status()
    client.errorEvent.emit(
        8,
        101,
        "Max number of market data subscriptions reached",
        contract,
    )
    capacity = connection.resource_status()

    assert pacing.pacing_violations_today == 1  # type: ignore[attr-defined]
    assert pacing.pacing_state == "VIOLATION_RECORDED"  # type: ignore[attr-defined]
    assert "DU***456" in pacing.last_resource_error  # type: ignore[operator, attr-defined]
    assert capacity.capacity_rejects_today == 1
    assert connection.is_connected is False


def option(con_id: int, *, right: str = "C") -> QualifiedOption:
    return QualifiedOption(
        symbol="AMD",
        con_id=con_id,
        exchange="SMART",
        currency="USD",
        security_type="OPT",
        expiry=date(2026, 9, 18),
        strike=100.0,
        right=right,
        multiplier="100",
        trading_class="AMD",
    )


def incomplete_ticker() -> SimpleNamespace:
    return SimpleNamespace(
        bid=float("nan"),
        ask=float("nan"),
        callOpenInterest=float("nan"),
        putOpenInterest=float("nan"),
        marketDataType=1,
        modelGreeks=None,
        histVolatility=float("nan"),
    )


def complete(ticker: SimpleNamespace, *, right: str = "C") -> None:
    ticker.bid = 1.0
    ticker.ask = 1.2
    ticker.callOpenInterest = 100.0 if right == "C" else float("nan")
    ticker.putOpenInterest = 100.0 if right == "P" else float("nan")
    ticker.modelGreeks = SimpleNamespace(impliedVol=0.25, delta=0.5, gamma=0.02)


def connected_stream_boundary(
    client: StreamClient, *, budget: int = 100, timeout: float = 0.2
) -> IbkrConnection:
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=92,
            expected_account="DU123456",
            request_timeout_seconds=timeout,
            market_data_line_budget=budget,
        ),
        client=client,  # type: ignore[arg-type]
    )
    connection._account_id = "DU123456"
    return connection


def test_active_stream_registry_increments_and_final_release_cancels() -> None:
    async def scenario() -> tuple[object, object]:
        client = StreamClient()
        connection = connected_stream_boundary(client)
        task = asyncio.create_task(connection.option_snapshots((option(101),)))
        await asyncio.sleep(0)
        active = connection.resource_status()
        complete(client.tickers[101])
        await task
        return active, connection.resource_status()

    active, released = asyncio.run(scenario())

    assert active.active_market_data_lines == 1  # type: ignore[attr-defined]
    assert active.active_option_lines == 1  # type: ignore[attr-defined]
    assert active.active_underlying_lines == 0  # type: ignore[attr-defined]
    assert active.subscriptions[0].con_id == 101  # type: ignore[attr-defined]
    assert active.subscriptions[0].purpose == "OPTION_PRE_CONTEXT"  # type: ignore[attr-defined]
    assert active.subscriptions[0].consumer_count == 1  # type: ignore[attr-defined]
    assert released.active_market_data_lines == 0  # type: ignore[attr-defined]


def test_historical_volatility_uses_only_temporary_generic_tick_104(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stocker_execution.ibkr._to_ib_contract",
        lambda instrument: SimpleNamespace(
            conId=instrument.con_id, secType="STK", exchange=instrument.exchange
        ),
    )

    async def scenario() -> tuple[StreamClient, object, object, object]:
        client = StreamClient()
        connection = connected_stream_boundary(client)
        task = asyncio.create_task(connection.historical_volatility_snapshot(stock(101)))
        await asyncio.sleep(0)
        active = connection.resource_status()
        client.tickers[101].histVolatility = 0.40
        result = await task
        return client, active, result, connection.resource_status()

    client, active, result, released = asyncio.run(scenario())

    assert client.generic_tick_lists == ["104"]
    assert client.cancelled == [101]
    assert active.active_market_data_lines == 1  # type: ignore[attr-defined]
    assert active.active_underlying_lines == 1  # type: ignore[attr-defined]
    assert active.active_option_lines == 0  # type: ignore[attr-defined]
    assert active.subscriptions[0].purpose == "SESSION_HARD_HV"  # type: ignore[attr-defined]
    assert result.raw_historical_volatility == 0.40
    assert result.unit == "DECIMAL"
    assert released.active_market_data_lines == 0  # type: ignore[attr-defined]


def test_invalid_historical_volatility_releases_subscription_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stocker_execution.ibkr._to_ib_contract",
        lambda instrument: SimpleNamespace(
            conId=instrument.con_id, secType="STK", exchange=instrument.exchange
        ),
    )
    client = StreamClient()
    connection = connected_stream_boundary(client, timeout=0.01)

    with pytest.raises(IbkrError, match="HV_NOT_READY"):
        asyncio.run(connection.historical_volatility_snapshot(stock(101)))

    assert client.generic_tick_lists == ["104"]
    assert client.cancelled == [101]
    assert connection.resource_status().active_market_data_lines == 0


def test_fifty_member_activity_watchlist_opens_no_streaming_quote() -> None:
    async def scenario() -> tuple[WatchlistScannerClient, int, object]:
        client = WatchlistScannerClient()
        connection = connected_stream_boundary(client)
        rows = {
            component: await connection.activity_scan(
                market=get_market(MarketId.US_NASDAQ),
                cap_bucket=CapBucket.MID,
                component=component,
                max_results=50,
            )
            for component in ActivityScanner
        }
        watchlist = rank_activity_candidates(rows)
        return client, len(watchlist), connection.resource_status()

    client, watchlist_size, status = asyncio.run(scenario())

    assert watchlist_size == 50
    assert client.requested == []
    assert status.active_market_data_lines == 0  # type: ignore[attr-defined]


def test_identical_physical_stream_is_shared_until_final_consumer_releases() -> None:
    async def scenario() -> tuple[StreamClient, object]:
        client = StreamClient()
        connection = connected_stream_boundary(client)
        first = asyncio.create_task(connection.option_snapshots((option(101),)))
        await asyncio.sleep(0)
        second = asyncio.create_task(connection.option_snapshots((option(101),)))
        await asyncio.sleep(0)
        shared = connection.resource_status()
        complete(client.tickers[101])
        await asyncio.gather(first, second)
        return client, shared

    client, shared = asyncio.run(scenario())

    assert client.requested == [101]
    assert client.cancelled == [101]
    assert shared.active_market_data_lines == 1  # type: ignore[attr-defined]
    assert shared.subscriptions[0].consumer_count == 2  # type: ignore[attr-defined]
    assert shared.deduplicated_requests_today == 1  # type: ignore[attr-defined]


def test_budget_exhaustion_rejects_noncritical_option_stream_without_disturbing_first() -> None:
    async def scenario() -> tuple[StreamClient, object]:
        client = StreamClient()
        connection = connected_stream_boundary(client, budget=1)
        first = asyncio.create_task(connection.option_snapshots((option(101),)))
        await asyncio.sleep(0)
        with pytest.raises(IbkrError, match="IBKR_MARKET_DATA_CAPACITY_UNAVAILABLE"):
            await connection.option_snapshots((option(202, right="P"),))
        rejected = connection.resource_status()
        complete(client.tickers[101])
        await first
        return client, rejected

    client, rejected = asyncio.run(scenario())

    assert client.requested == [101]
    assert rejected.active_market_data_lines == 1  # type: ignore[attr-defined]
    assert rejected.capacity_rejects_today == 1  # type: ignore[attr-defined]
    assert rejected.last_resource_error == "IBKR_MARKET_DATA_CAPACITY_UNAVAILABLE"  # type: ignore[attr-defined]


def test_timeout_and_exception_release_only_streams_that_were_opened() -> None:
    timeout_client = StreamClient()
    timeout_connection = connected_stream_boundary(timeout_client, timeout=0.01)
    asyncio.run(timeout_connection.option_snapshots((option(101),)))
    assert timeout_connection.resource_status().active_market_data_lines == 0  # type: ignore[attr-defined]
    assert timeout_client.cancelled == [101]

    failing_client = StreamClient()
    failing_client.raise_for.add(202)
    failing_connection = connected_stream_boundary(failing_client)
    with pytest.raises(IbkrError, match="market data request failed"):
        asyncio.run(failing_connection.option_snapshots((option(101), option(202, right="P"))))
    assert failing_connection.resource_status().active_market_data_lines == 0  # type: ignore[attr-defined]
    assert failing_client.requested == [101]
    assert failing_client.cancelled == [101]


def stock(con_id: int) -> QualifiedInstrument:
    return QualifiedInstrument(
        symbol=f"STK{con_id}",
        con_id=con_id,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
    )


async def wait_until(value: object, expected: int) -> None:
    for _ in range(100):
        if int(value()) == expected:  # type: ignore[operator]
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {expected}")


def test_historical_request_concurrency_is_bounded_without_changing_bar_semantics() -> None:
    async def scenario() -> tuple[ConcurrencyClient, object]:
        client = ConcurrencyClient()
        connection = connected_stream_boundary(client)
        tasks = [
            asyncio.create_task(
                connection.historical_bars(
                    stock(index),
                    bar_size="1 min",
                    duration="300 S",
                    what_to_show="TRADES",
                    regular_trading_hours=True,
                )
            )
            for index in range(10)
        ]
        await wait_until(lambda: client.active_history, 4)
        active = connection.resource_status()
        client.history_gate.set()
        await asyncio.gather(*tasks)
        return client, active

    client, active = asyncio.run(scenario())

    assert client.peak_history == 4
    assert active.pending_historical_work == 10  # type: ignore[attr-defined]
    assert active.historical_concurrency_limit == 4  # type: ignore[attr-defined]


def test_scanner_concurrency_never_exceeds_ten_and_completed_scans_are_released() -> None:
    async def scenario() -> tuple[ConcurrencyClient, object, object]:
        client = ConcurrencyClient()
        connection = connected_stream_boundary(client)
        await connection.scanner_capabilities()
        tasks = [
            asyncio.create_task(
                connection.activity_scan(
                    market=get_market(MarketId.US_NASDAQ),
                    cap_bucket=CapBucket.MID,
                    component=ActivityScanner.TOP_TRADE_RATE,
                )
            )
            for _ in range(11)
        ]
        await wait_until(lambda: client.active_scanners, 10)
        active = connection.resource_status()
        client.scanner_gate.set()
        await asyncio.gather(*tasks)
        return client, active, connection.resource_status()

    client, active, released = asyncio.run(scenario())

    assert client.peak_scanners == 10
    assert client.parameter_requests == 1
    assert active.active_scanners == 10  # type: ignore[attr-defined]
    assert released.active_scanners == 0  # type: ignore[attr-defined]
    assert released.scanner_requests_today == 11  # type: ignore[attr-defined]


def test_scanner_timeout_cancels_broker_subscription_and_clears_registry() -> None:
    async def scenario() -> tuple[LowLevelScannerClient, object]:
        client = LowLevelScannerClient()
        connection = connected_stream_boundary(client, timeout=0.01)
        with pytest.raises(IbkrError, match="scanner request failed"):
            await connection.hot_us_stocks_by_volume()
        return client, connection.resource_status()

    client, status = asyncio.run(scenario())

    assert client.cancelled_scanners == [71]
    assert status.active_scanners == 0  # type: ignore[attr-defined]


def test_daily_resource_counters_roll_over() -> None:
    client = ErrorClient()
    connection = IbkrConnection(config(), client=client)  # type: ignore[arg-type]
    client.errorEvent.emit(7, 100, "pacing violation", None)
    assert connection.resource_status().pacing_violations_today == 1

    connection._reset_daily_resource_counters(connection._resource_counter_date + timedelta(days=1))

    assert connection._pacing_violations_today == 0
    assert connection._market_data_requests_today == 0


def test_contract_qualification_and_option_definitions_reuse_session_cache() -> None:
    async def scenario() -> tuple[CacheClient, object]:
        client = CacheClient()
        connection = connected_stream_boundary(client)
        first = await connection.resolve_stock("aapl", exchange="smart", currency="usd")
        second = await connection.resolve_stock("AAPL", exchange="SMART", currency="USD")
        first_chain = await connection.option_chains(first)
        second_chain = await connection.option_chains(second)
        assert first == second
        assert first_chain is second_chain
        return client, connection.resource_status()

    client, status = asyncio.run(scenario())

    assert client.qualification_requests == 1
    assert client.option_chain_requests == 1
    assert status.deduplicated_requests_today == 2  # type: ignore[attr-defined]


def test_disconnect_discards_twenty_stale_streams_and_reconnect_does_not_double() -> None:
    async def scenario() -> tuple[CacheClient, object, object, list[int]]:
        client = CacheClient()
        connection = connected_stream_boundary(client, timeout=5.0)
        capture = asyncio.create_task(
            connection.option_snapshots(tuple(option(index) for index in range(1, 21)))
        )
        await wait_until(
            lambda: connection.resource_status().active_market_data_lines,  # type: ignore[attr-defined]
            20,
        )
        before = connection.resource_status()
        connection.disconnect()
        after_disconnect = connection.resource_status()
        assert after_disconnect.active_market_data_lines == 0  # type: ignore[attr-defined]
        await connection.connect()
        after_reconnect = connection.resource_status()
        capture.cancel()
        with suppress(asyncio.CancelledError):
            await capture
        return client, before, after_reconnect, client.cancelled

    client, before, after, cancelled = asyncio.run(scenario())

    assert before.active_market_data_lines == 20  # type: ignore[attr-defined]
    assert after.active_market_data_lines == 0  # type: ignore[attr-defined]
    assert client.requested == list(range(1, 21))
    assert cancelled == list(range(1, 21))


def test_realistic_capacity_simulation_shares_physical_work_and_stays_below_budget() -> None:
    scenarios = {item.name: item for item in simulate_capacity(market_data_line_budget=100)}

    assert scenarios["A"].configured_runs == 1
    assert scenarios["A"].unique_stocks == 50
    assert scenarios["A"].scanner_requests == 3
    assert scenarios["A"].peak_market_data_lines == 2
    assert scenarios["B"].configured_runs == 3
    assert scenarios["B"].unique_stocks == 50
    assert scenarios["B"].duplicate_stock_requests_avoided == 100
    assert scenarios["B"].scanner_requests == 3
    assert scenarios["C"].configured_runs == 4
    assert scenarios["C"].unique_stocks == 200
    assert scenarios["C"].scanner_requests == 12
    assert scenarios["C"].peak_scanner_concurrency == 1
    assert scenarios["C"].historical_requests == 6_200
    assert scenarios["D"].subscriptions_before_disconnect == 20
    assert scenarios["D"].subscriptions_after_disconnect == 0
    assert scenarios["D"].subscriptions_after_reconnect == 0
    realistic = scenarios["REALISTIC_4_RUN"]
    assert realistic.configured_runs == 4
    assert realistic.unique_stocks == 100
    assert realistic.contract_qualifications == 100
    assert realistic.stage4_contexts == 100
    assert realistic.historical_requests == 3_100
    assert realistic.peak_underlying_lines == 0
    assert realistic.peak_option_lines == 2
    assert realistic.peak_market_data_lines == 2
    assert realistic.within_budget is True

    exhausted = {item.name: item for item in simulate_capacity(market_data_line_budget=1)}["A"]
    assert exhausted.within_budget is False
    assert exhausted.unique_stocks == 50
    assert exhausted.stage4_contexts == 0
    assert exhausted.historical_requests == 50
    assert exhausted.execution_safety_checks == 50


def test_read_only_ibkr_resource_diagnostic_reports_configured_workload(tmp_path: Path) -> None:
    runs = tmp_path / "runs.yaml"
    brokers = tmp_path / "ibkr.yaml"
    runs.write_text(
        """
universes:
  - universe_id: CUSTOM
    name: Custom
    members:
      - {symbol: AAPL, exchange: SMART, primary_exchange: NASDAQ, currency: USD}
      - {symbol: MSFT, exchange: SMART, primary_exchange: NASDAQ, currency: USD}
runs:
  - run_id: first
    universe: CUSTOM
    strategy: SESSION_HARD
    environment: PAPER
  - run_id: second
    universe: CUSTOM
    strategy: SESSION_HARD
    environment: LIVE
""",
        encoding="utf-8",
    )
    brokers.write_text(
        """
PAPER:
  environment: PAPER
  host: 127.0.0.1
  port: 4002
  client_id: 91
  expected_account: DU123456
  market_data_line_budget: 80
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ibkr-resources",
            "--runs-config",
            str(runs),
            "--ibkr-config",
            str(brokers),
            "--environment",
            "PAPER",
        ],
    )

    assert result.exit_code == 0
    assert "Stocker API line budget: 80" in result.stdout
    assert "Diagnostic connection active streaming lines: 0" in result.stdout
    assert "first: 2 candidates (configured universe)" in result.stdout
    assert "second: 2 candidates (configured universe)" in result.stdout
    assert "Unique configured physical identities: 2" in result.stdout
    assert "Estimated duplicate savings: 2" in result.stdout
    assert "ib_async throttle: 40 requests / 1" in result.stdout
    assert "No order was transmitted" in result.stdout


def test_resource_diagnostic_never_invents_activity_watchlist_members(tmp_path: Path) -> None:
    runs = tmp_path / "runs.yaml"
    brokers = tmp_path / "ibkr.yaml"
    database = tmp_path / "absent.sqlite3"
    runs.write_text(
        """
universes:
  - universe_id: NASDAQ_MID
    name: NASDAQ mid-cap screen
    market_spec: {market_id: US_NASDAQ, cap_bucket: MID, cap_bucket_version: CAP_BUCKETS_V1}
runs:
  - run_id: activity
    universe: NASDAQ_MID
    strategy: SESSION_HARD
    strategy_id: SESSION_HARD_HIGH_PRE_MOVE_DOWN_STRUCTURE_D
    strategy_version: SESSION_HARD_STRUCTURE_D_V1
    market_id: US_NASDAQ
    cap_bucket: MID
    cap_bucket_version: CAP_BUCKETS_V1
    candidate_screen_id: ACTIVITY_SHORTLIST_V1
    candidate_screen_version: ACTIVITY_SHORTLIST_V1
    display_name: NASDAQ · HARD · MID
    environment: PAPER
    screen:
      method: ACTIVITY_SHORTLIST_V1
      max_results: 50
      version: ACTIVITY_SHORTLIST_V1
      scheduled_active_minutes: 15
""",
        encoding="utf-8",
    )
    brokers.write_text(
        """
PAPER:
  environment: PAPER
  host: 127.0.0.1
  port: 4002
  client_id: 91
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ibkr-resources",
            "--runs-config",
            str(runs),
            "--ibkr-config",
            str(brokers),
            "--database",
            str(database),
        ],
    )

    assert result.exit_code == 0
    assert "activity: unavailable (no persisted Activity Shortlist)" in result.stdout
    assert "Estimated duplicate savings: unavailable" in result.stdout
    assert "activity: 50 candidates" not in result.stdout
    assert not database.exists()


def test_resource_diagnostic_reads_actual_frozen_activity_members_read_only(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runtime.sqlite3"
    ActivityShortlistStore(database).save_once(
        ActivityShortlistSnapshot(
            market_id="US_NASDAQ",
            cap_bucket=CapBucket.MID,
            cap_bucket_version="CAP_BUCKETS_V1",
            session=date(2026, 9, 2),
            screen_timestamp=datetime(2026, 9, 2, 13, 45, tzinfo=UTC),
            profile_id="ACTIVITY_SHORTLIST_V1",
            profile_version="ACTIVITY_SHORTLIST_V1",
            status=ActivityShortlistStatus.READY,
            components=(ActivityScanner.TOP_TRADE_RATE,),
            candidates=(
                ActivityCandidate(
                    symbol="AAPL",
                    con_id=265598,
                    exchange="SMART",
                    primary_exchange="NASDAQ",
                    currency="USD",
                    top_trade_rate_rank=1,
                    top_volume_rate_rank=None,
                    hot_by_volume_rank=None,
                    scan_hit_count=1,
                    best_component_rank=1,
                    aggregate_screen_score=1.0,
                    final_shortlist_rank=1,
                    selected=True,
                ),
            ),
        )
    )

    watchlist = ActivityShortlistStore.latest_read_only(
        database,
        market_id="US_NASDAQ",
        cap_bucket=CapBucket.MID,
        cap_bucket_version="CAP_BUCKETS_V1",
        profile_id="ACTIVITY_SHORTLIST_V1",
        profile_version="ACTIVITY_SHORTLIST_V1",
    )

    assert watchlist is not None
    assert watchlist.status is ActivityShortlistStatus.READY
    assert tuple(item.con_id for item in watchlist.candidates) == (265598,)
