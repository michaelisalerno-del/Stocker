import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from stocker_core.config import IbkrConfig
from stocker_core.runs import Environment
from stocker_execution.history import (
    HistorySemantics,
    HistoryStatus,
    IbkrHistoryCache,
    IbkrHistoryService,
)
from stocker_execution.ibkr import (
    HistoricalBar,
    IbkrConnection,
    IbkrError,
    QualifiedInstrument,
)


class FakeIbClient:
    def __init__(self, result: list[object] | Exception) -> None:
        self.result = result
        self.connected = False
        self.requests = 0

    async def connectAsync(self, *args: object, **kwargs: object) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def isConnected(self) -> bool:
        return self.connected

    def managedAccounts(self) -> list[str]:
        return ["DU123456"]

    async def qualifyContractsAsync(
        self, *contracts: object, returnAll: bool = False
    ) -> list[object]:
        return []

    async def reqHistoricalDataAsync(
        self, contract: object, **kwargs: object
    ) -> list[object]:
        self.requests += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    async def reqTickersAsync(
        self, *contracts: object, regulatorySnapshot: bool = False
    ) -> list[object]:
        return []


def source_bar(value: HistoricalBar) -> object:
    return SimpleNamespace(
        date=value.timestamp,
        open=value.open,
        high=value.high,
        low=value.low,
        close=value.close,
        volume=value.volume,
    )


def connected_ibkr(
    result: list[object] | Exception,
) -> tuple[IbkrConnection, FakeIbClient]:
    client = FakeIbClient(result)
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=41,
        ),
        client=client,
    )
    asyncio.run(connection.connect())
    return connection, client


def seed_cache(
    cache: IbkrHistoryCache,
    target: QualifiedInstrument,
    semantics: HistorySemantics,
    bars: tuple[HistoricalBar, ...],
) -> FakeIbClient:
    connection, client = connected_ibkr([source_bar(item) for item in bars])
    service = IbkrHistoryService(connection, cache)
    asyncio.run(
        service.fetch_and_store(
            target,
            bar_size=semantics.bar_size,
            duration="1 D",
            what_to_show=semantics.what_to_show,
            regular_trading_hours=semantics.regular_trading_hours,
        )
    )
    return client


def instrument(*, con_id: int = 4391, symbol: str = "AMD") -> QualifiedInstrument:
    return QualifiedInstrument(
        symbol=symbol,
        con_id=con_id,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
    )


def bar(timestamp: datetime, price: float) -> HistoricalBar:
    return HistoricalBar(
        timestamp=timestamp,
        open=price,
        high=price + 0.2,
        low=price - 0.1,
        close=price + 0.1,
        volume=1_000.0,
    )


def test_same_conid_reuses_history_across_consumers_without_ticker_keying(
    tmp_path: Path,
) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    semantics = HistorySemantics("1 min", "TRADES", regular_trading_hours=False)
    timestamp = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    seed_cache(cache, instrument(), semantics, (bar(timestamp, 150.0),))

    same_contract_from_another_consumer = instrument(symbol="AMD.QUALIFIED")
    snapshot = cache.get_required_history(
        same_contract_from_another_consumer,
        semantics,
        required_timestamps=(timestamp,),
        as_of=timestamp,
    )

    assert snapshot.status is HistoryStatus.READY
    assert snapshot.con_id == 4391
    assert snapshot.source == "IBKR"
    assert snapshot.bars == (bar(timestamp, 150.0),)


def test_ticker_alone_cannot_mix_distinct_qualified_contracts(tmp_path: Path) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    semantics = HistorySemantics("1 min", "TRADES", regular_trading_hours=False)
    timestamp = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    seed_cache(cache, instrument(con_id=1), semantics, (bar(timestamp, 100.0),))
    seed_cache(cache, instrument(con_id=2), semantics, (bar(timestamp, 200.0),))

    first = cache.get_required_history(
        instrument(con_id=1), semantics, (timestamp,), as_of=timestamp
    )
    second = cache.get_required_history(
        instrument(con_id=2), semantics, (timestamp,), as_of=timestamp
    )

    assert first.bars[0].open == 100.0
    assert second.bars[0].open == 200.0


def test_request_semantics_are_not_interchangeable(tmp_path: Path) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    timestamp = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    trades = HistorySemantics("1 min", "TRADES", regular_trading_hours=False)
    midpoint = HistorySemantics("5 mins", "MIDPOINT", regular_trading_hours=True)
    seed_cache(cache, instrument(), trades, (bar(timestamp, 150.0),))

    snapshot = cache.get_required_history(instrument(), midpoint, (timestamp,), as_of=timestamp)

    assert snapshot.status is HistoryStatus.NOT_READY
    assert snapshot.bars == ()
    assert snapshot.missing_timestamps == (timestamp,)
    assert "incomplete IBKR history" in snapshot.reason


def test_incomplete_history_is_not_interpolated_or_silently_reduced(
    tmp_path: Path,
) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    semantics = HistorySemantics("1 min", "TRADES", regular_trading_hours=False)
    start = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    required = tuple(start + timedelta(minutes=index) for index in range(3))
    seed_cache(
        cache,
        instrument(),
        semantics,
        (bar(required[0], 150.0), bar(required[2], 152.0)),
    )

    snapshot = cache.get_required_history(instrument(), semantics, required, as_of=required[-1])

    assert snapshot.status is HistoryStatus.NOT_READY
    assert tuple(item.timestamp for item in snapshot.bars) == (required[0], required[2])
    assert snapshot.missing_timestamps == (required[1],)


def test_invalid_cached_bar_is_rejected_as_not_ready(tmp_path: Path) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    semantics = HistorySemantics("1 min", "TRADES", regular_trading_hours=False)
    timestamp = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    seed_cache(cache, instrument(), semantics, (bar(timestamp, 150.0),))
    with sqlite3.connect(cache.path) as connection:
        connection.execute("UPDATE ibkr_history_bars SET high = 1.0")

    snapshot = cache.get_required_history(
        instrument(), semantics, (timestamp,), as_of=timestamp
    )

    assert snapshot.status is HistoryStatus.NOT_READY
    assert snapshot.bars == ()
    assert snapshot.missing_timestamps == (timestamp,)
    assert "invalid cached IBKR history" in snapshot.reason


def test_future_cached_bars_are_excluded_by_as_of(tmp_path: Path) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    semantics = HistorySemantics("1 min", "TRADES", regular_trading_hours=False)
    start = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    timestamps = tuple(start + timedelta(minutes=index) for index in range(3))
    seed_cache(
        cache,
        instrument(),
        semantics,
        tuple(bar(timestamp, 150.0 + index) for index, timestamp in enumerate(timestamps)),
    )

    snapshot = cache.get_required_history(
        instrument(), semantics, timestamps, as_of=timestamps[1]
    )

    assert snapshot.status is HistoryStatus.NOT_READY
    assert tuple(item.timestamp for item in snapshot.bars) == timestamps[:2]
    assert snapshot.missing_timestamps == (timestamps[2],)


def test_cache_requires_timezone_aware_timestamps(tmp_path: Path) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    semantics = HistorySemantics("1 min", "TRADES", regular_trading_hours=False)
    naive = datetime(2026, 9, 1, 13, 30)

    with pytest.raises(ValueError, match="timezone-aware"):
        seed_cache(cache, instrument(), semantics, (bar(naive, 150.0),))


def test_cache_has_no_run_environment_or_alternative_provider_dimension(
    tmp_path: Path,
) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    semantics = HistorySemantics("1 min", "trades", regular_trading_hours=False)
    timestamp = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    seed_cache(cache, instrument(), semantics, (bar(timestamp, 150.0),))

    snapshot = cache.get_required_history(instrument(), semantics, (timestamp,), as_of=timestamp)

    assert snapshot.status is HistoryStatus.READY
    assert snapshot.source == "IBKR"
    assert "environment" not in semantics.__dataclass_fields__
    assert "source" not in semantics.__dataclass_fields__


def test_history_service_persists_only_stage2_ibkr_results(tmp_path: Path) -> None:
    timestamp = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    boundary, client = connected_ibkr([source_bar(bar(timestamp, 150.0))])
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    service = IbkrHistoryService(boundary, cache)

    async def scenario() -> None:
        await service.fetch_and_store(
            instrument(),
            bar_size="1 min",
            duration="60 S",
            what_to_show="TRADES",
            regular_trading_hours=False,
            end_time=timestamp,
        )

    asyncio.run(scenario())
    snapshot = cache.get_required_history(
        instrument(),
        HistorySemantics("1 min", "TRADES", regular_trading_hours=False),
        (timestamp,),
        as_of=timestamp,
    )

    assert client.requests == 1
    assert snapshot.status is HistoryStatus.READY
    assert snapshot.source == "IBKR"


def test_history_ingress_rejects_non_ibkr_adapters(tmp_path: Path) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")

    with pytest.raises(TypeError, match="concrete Stage 2 IbkrConnection"):
        IbkrHistoryService(object(), cache)  # type: ignore[arg-type]

    assert not hasattr(cache, "store")


def test_ibkr_request_failure_leaves_history_not_ready(tmp_path: Path) -> None:
    timestamp = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    boundary, _ = connected_ibkr(RuntimeError("request rejected"))
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    service = IbkrHistoryService(boundary, cache)

    async def scenario() -> None:
        await service.fetch_and_store(
            instrument(),
            bar_size="1 min",
            duration="60 S",
            what_to_show="TRADES",
            regular_trading_hours=False,
            end_time=timestamp,
        )

    with pytest.raises(IbkrError, match="request rejected"):
        asyncio.run(scenario())

    snapshot = cache.get_required_history(
        instrument(),
        HistorySemantics("1 min", "TRADES", regular_trading_hours=False),
        (timestamp,),
        as_of=timestamp,
    )
    assert snapshot.status is HistoryStatus.NOT_READY
    assert snapshot.bars == ()
