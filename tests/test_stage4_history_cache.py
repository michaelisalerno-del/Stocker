import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stocker_execution.history import (
    HistorySemantics,
    HistoryStatus,
    IbkrHistoryCache,
    IbkrHistoryService,
)
from stocker_execution.ibkr import HistoricalBar, IbkrError, QualifiedInstrument


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
    cache.store(instrument(), semantics, (bar(timestamp, 150.0),))

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
    cache.store(instrument(con_id=1), semantics, (bar(timestamp, 100.0),))
    cache.store(instrument(con_id=2), semantics, (bar(timestamp, 200.0),))

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
    cache.store(instrument(), trades, (bar(timestamp, 150.0),))

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
    cache.store(instrument(), semantics, (bar(required[0], 150.0), bar(required[2], 152.0)))

    snapshot = cache.get_required_history(instrument(), semantics, required, as_of=required[-1])

    assert snapshot.status is HistoryStatus.NOT_READY
    assert tuple(item.timestamp for item in snapshot.bars) == (required[0], required[2])
    assert snapshot.missing_timestamps == (required[1],)


def test_future_cached_bars_are_excluded_by_as_of(tmp_path: Path) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    semantics = HistorySemantics("1 min", "TRADES", regular_trading_hours=False)
    start = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    timestamps = tuple(start + timedelta(minutes=index) for index in range(3))
    cache.store(
        instrument(),
        semantics,
        tuple(bar(timestamp, 150.0 + index) for index, timestamp in enumerate(timestamps)),
    )

    snapshot = cache.get_required_history(
        instrument(), semantics, timestamps[:2], as_of=timestamps[1]
    )

    assert snapshot.status is HistoryStatus.READY
    assert tuple(item.timestamp for item in snapshot.bars) == timestamps[:2]


def test_cache_requires_timezone_aware_timestamps(tmp_path: Path) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    semantics = HistorySemantics("1 min", "TRADES", regular_trading_hours=False)
    naive = datetime(2026, 9, 1, 13, 30)

    with pytest.raises(ValueError, match="timezone-aware"):
        cache.store(instrument(), semantics, (bar(naive, 150.0),))


def test_cache_has_no_run_environment_or_alternative_provider_dimension(
    tmp_path: Path,
) -> None:
    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    semantics = HistorySemantics("1 min", "trades", regular_trading_hours=False)
    timestamp = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    cache.store(instrument(), semantics, (bar(timestamp, 150.0),))

    snapshot = cache.get_required_history(instrument(), semantics, (timestamp,), as_of=timestamp)

    assert snapshot.status is HistoryStatus.READY
    assert snapshot.source == "IBKR"
    assert "environment" not in semantics.__dataclass_fields__
    assert "source" not in semantics.__dataclass_fields__


def test_history_service_persists_only_stage2_ibkr_results(tmp_path: Path) -> None:
    timestamp = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)

    class Stage2Boundary:
        calls = 0

        async def historical_bars(
            self, *args: object, **kwargs: object
        ) -> tuple[HistoricalBar, ...]:
            self.calls += 1
            return (bar(timestamp, 150.0),)

    boundary = Stage2Boundary()
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

    assert boundary.calls == 1
    assert snapshot.status is HistoryStatus.READY
    assert snapshot.source == "IBKR"


def test_ibkr_request_failure_leaves_history_not_ready(tmp_path: Path) -> None:
    timestamp = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)

    class FailingStage2Boundary:
        async def historical_bars(
            self, *args: object, **kwargs: object
        ) -> tuple[HistoricalBar, ...]:
            raise IbkrError("IBKR request rejected")

    cache = IbkrHistoryCache(tmp_path / "ibkr-history.sqlite3")
    service = IbkrHistoryService(FailingStage2Boundary(), cache)

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
