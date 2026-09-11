from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_core.markets import MarketDefinition
from stocker_execution.expected_move import ExpectedMoveResult, ExpectedMoveStatus
from stocker_execution.history import IbkrHistoryCache
from stocker_execution.ibkr import HistoricalBar, IbkrConnection, QualifiedInstrument
from stocker_execution.stage5 import (
    STAGE5_HV_CALCULATION_VERSION,
    Stage5Analyzer,
    Stage5CurrentDataService,
    Stage5Membership,
    Stage5QualifiedRequest,
    Stage5Status,
)

T0 = datetime(2025, 2, 20, 15, 0, tzinfo=UTC)


def bar(timestamp: datetime, open_price: float) -> HistoricalBar:
    return HistoricalBar(timestamp, open_price, open_price, open_price, open_price, 100.0)


def instrument() -> QualifiedInstrument:
    return QualifiedInstrument("HOOD", 123, "SMART", "NASDAQ", "USD", "STK")


class HistoryBoundary(IbkrConnection):
    def __init__(self, *, missing_minus_3m: bool = False) -> None:
        self.calls: list[str] = []
        self.missing_minus_3m = missing_minus_3m

    async def historical_bars(
        self,
        requested: QualifiedInstrument,
        *,
        bar_size: str,
        duration: str,
        what_to_show: str,
        regular_trading_hours: bool,
        end_time: date | datetime | None = None,
        minimum_bars: int = 1,
    ) -> tuple[HistoricalBar, ...]:
        del requested, duration, end_time, minimum_bars
        assert what_to_show == "TRADES"
        assert regular_trading_hours is True
        self.calls.append(bar_size)
        if bar_size == "5 mins":
            return (bar(T0, 100.0),)
        rows = [bar(T0, 100.0)]
        if not self.missing_minus_3m:
            rows.insert(0, bar(T0 - timedelta(minutes=3), 99.0))
        return tuple(rows)


class ContextService:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready

    async def get_expected_move(
        self,
        requested: QualifiedInstrument,
        *,
        market: MarketDefinition,
        session: date,
        t0: datetime,
    ) -> ExpectedMoveResult:
        del requested, market, session, t0
        if not self.ready:
            return ExpectedMoveResult(
                status=ExpectedMoveStatus.NOT_READY,
                expected_absolute_return_15m=None,
                source="IBKR_HISTORICAL_VOLATILITY_TICK_104",
                observation_timestamp=None,
                calculation_version="EXPECTED_MOVE_HV_V1",
                reason="PRE_CONTEXT_NOT_READY: missing frozen context",
            )
        return ExpectedMoveResult(
            status=ExpectedMoveStatus.READY,
            expected_absolute_return_15m=0.01,
            source="IBKR_HISTORICAL_VOLATILITY_TICK_104",
            observation_timestamp=T0 - timedelta(days=1),
            calculation_version="EXPECTED_MOVE_HV_V1",
            reason="cache hit",
        )


def test_current_data_service_fetches_only_ibkr_5m_and_1m_inputs_and_reuses_cache(
    tmp_path: Path,
) -> None:
    boundary = HistoryBoundary()
    service = Stage5CurrentDataService(
        boundary,
        IbkrHistoryCache(tmp_path / "history.sqlite3"),
        ContextService(),
        clock=lambda: T0 + timedelta(minutes=5),
    )

    first = asyncio.run(service.get_feature(instrument(), session=T0.date(), t0=T0))
    second = asyncio.run(service.get_feature(instrument(), session=T0.date(), t0=T0))

    assert first.status is Stage5Status.READY
    assert second == first
    assert boundary.calls == ["5 mins", "1 min"]


def test_missing_exact_ibkr_minute_is_not_ready(tmp_path: Path) -> None:
    service = Stage5CurrentDataService(
        HistoryBoundary(missing_minus_3m=True),
        IbkrHistoryCache(tmp_path / "history.sqlite3"),
        ContextService(),
        clock=lambda: T0 + timedelta(minutes=5),
    )

    result = asyncio.run(service.get_feature(instrument(), session=T0.date(), t0=T0))

    assert result.status is Stage5Status.PRE_MOVE_NOT_READY
    assert "T0-3m" in result.exclusion_reason


def test_missing_stage4_context_does_not_request_current_bars(tmp_path: Path) -> None:
    boundary = HistoryBoundary()
    service = Stage5CurrentDataService(
        boundary,
        IbkrHistoryCache(tmp_path / "history.sqlite3"),
        ContextService(ready=False),
        clock=lambda: T0 + timedelta(minutes=5),
    )

    result = asyncio.run(service.get_feature(instrument(), session=T0.date(), t0=T0))

    assert result.status is Stage5Status.PRE_CONTEXT_NOT_READY
    assert boundary.calls == []


class HvExpectedMoveService:
    async def get_expected_move(
        self,
        requested: QualifiedInstrument,
        *,
        market: MarketDefinition,
        session: date,
        t0: datetime,
    ) -> ExpectedMoveResult:
        del requested, session
        return ExpectedMoveResult(
            status=ExpectedMoveStatus.READY,
            expected_absolute_return_15m=0.00333310044188278,
            source="IBKR_HISTORICAL_VOLATILITY_TICK_104",
            observation_timestamp=t0,
            calculation_version="EXPECTED_MOVE_HV_V1",
            reason="",
            raw_historical_volatility=0.40,
            historical_volatility=0.40,
            market_regular_minutes=market.active_regular_minutes,
        )


class CapturingExpectedMovePreparation(HvExpectedMoveService):
    def __init__(self) -> None:
        self.con_ids: list[int] = []

    async def prepare_expected_move(
        self,
        requested: QualifiedInstrument,
        *,
        market: MarketDefinition,
        session: date,
        t0: datetime,
    ) -> None:
        del market, session, t0
        self.con_ids.append(requested.con_id)


def test_expected_move_preparation_isolates_one_unmappable_instrument(tmp_path: Path) -> None:
    expected_move = CapturingExpectedMovePreparation()
    service = Stage5CurrentDataService(
        HistoryBoundary(),
        IbkrHistoryCache(tmp_path / "history.sqlite3"),
        expected_move,
    )
    bad = QualifiedInstrument("BAD", 456, "SMART", "UNKNOWN", "ZZZ", "STK")

    asyncio.run(
        service.prepare_expected_moves(
            (
                Stage5QualifiedRequest(
                    instrument(),
                    (Stage5Membership("HV_RUN", "US_ALL"),),
                ),
                Stage5QualifiedRequest(
                    bad,
                    (Stage5Membership("HV_RUN", "UNKNOWN"),),
                ),
            ),
            session=T0.date(),
            t0=T0,
        )
    )

    assert expected_move.con_ids == [123]


def test_cached_universe_preparation_leaves_turns_for_dashboard_requests(tmp_path):
    from dataclasses import replace

    expected_move = CapturingExpectedMovePreparation()
    service = Stage5CurrentDataService(
        HistoryBoundary(), IbkrHistoryCache(tmp_path / "history.sqlite3"), expected_move
    )
    requests = tuple(
        Stage5QualifiedRequest(
            replace(instrument(), con_id=con_id), (Stage5Membership("HV_RUN", "US_ALL"),)
        )
        for con_id in range(1, 65)
    )
    observed = []

    async def scenario():
        loop = asyncio.get_running_loop()
        finished = False

        def dashboard_turn():
            if not finished:
                observed.append(len(expected_move.con_ids))
                loop.call_soon(dashboard_turn)

        loop.call_soon(dashboard_turn)
        await service.prepare_expected_moves(requests, session=T0.date(), t0=T0)
        finished = True

    asyncio.run(scenario())
    assert sorted(expected_move.con_ids) == list(range(1, 65))
    assert any(0 < count < 64 for count in observed), observed


def test_hv_stage5_reuses_pre_move_arithmetic_with_distinct_lineage(tmp_path: Path) -> None:
    result = asyncio.run(
        Stage5CurrentDataService(
            HistoryBoundary(),
            IbkrHistoryCache(tmp_path / "history.sqlite3"),
            HvExpectedMoveService(),
            calculation_version=STAGE5_HV_CALCULATION_VERSION,
            clock=lambda: T0 + timedelta(minutes=5),
        ).get_feature(instrument(), session=T0.date(), t0=T0)
    )

    assert result.status is Stage5Status.READY
    assert result.calculation_version == "STAGE5_PRE_MOVE_HV_V1"
    assert result.expected_move_source == "IBKR_HISTORICAL_VOLATILITY_TICK_104"
    assert result.expected_move_observation_at == T0
    assert result.expected_move_calculation_version == "EXPECTED_MOVE_HV_V1"
    assert result.raw_historical_volatility == 0.40
    assert result.historical_volatility == 0.40
    assert result.market_regular_minutes == 390
    assert result.m_price == pytest.approx(0.333310044188278)
    assert result.pre_move_m == pytest.approx(3.000209182540385)


class MixedHvExpectedMoveService(HvExpectedMoveService):
    async def get_expected_move(
        self,
        requested: QualifiedInstrument,
        *,
        market: MarketDefinition,
        session: date,
        t0: datetime,
    ) -> ExpectedMoveResult:
        if requested.symbol == "BAD":
            return ExpectedMoveResult(
                status=ExpectedMoveStatus.NOT_READY,
                expected_absolute_return_15m=None,
                source="IBKR_HISTORICAL_VOLATILITY_TICK_104",
                observation_timestamp=None,
                calculation_version="EXPECTED_MOVE_HV_V1",
                reason="HV_NOT_READY: missing, invalid, or unavailable tick 104",
            )
        return await super().get_expected_move(requested, market=market, session=session, t0=t0)


def test_one_missing_hv_rejects_only_that_symbol_and_batch_continues(tmp_path: Path) -> None:
    boundary = HistoryBoundary()
    analyzer = Stage5Analyzer(
        Stage5CurrentDataService(
            boundary,
            IbkrHistoryCache(tmp_path / "history.sqlite3"),
            MixedHvExpectedMoveService(),
            calculation_version=STAGE5_HV_CALCULATION_VERSION,
            clock=lambda: T0 + timedelta(minutes=5),
        )
    )
    requests = tuple(
        Stage5QualifiedRequest(
            QualifiedInstrument(symbol, con_id, "SMART", "NASDAQ", "USD", "STK"),
            (Stage5Membership("HV_RUN", "US_ALL"),),
        )
        for symbol, con_id in (("GOOD", 123), ("BAD", 456))
    )

    rows = asyncio.run(analyzer.analyze(requests, session=T0.date(), t0=T0))

    assert [(row.symbol, row.status, row.exclusion_reason) for row in rows] == [
        ("GOOD", Stage5Status.READY, ""),
        (
            "BAD",
            Stage5Status.PRE_CONTEXT_NOT_READY,
            "HV_NOT_READY: missing, invalid, or unavailable tick 104",
        ),
    ]
    assert boundary.calls == ["5 mins", "1 min"]
