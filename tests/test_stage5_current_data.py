from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from stocker_execution.history import IbkrHistoryCache
from stocker_execution.ibkr import HistoricalBar, IbkrConnection, QualifiedInstrument
from stocker_execution.pre_context import ContextStatus
from stocker_execution.stage5 import CandidateStatus, Stage5CurrentDataService

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

    async def get_or_create(self, requested: QualifiedInstrument, *, session: date) -> object:
        del requested, session
        if not self.ready:
            return SimpleNamespace(
                status=ContextStatus.NOT_READY,
                context=None,
                reason="PRE_CONTEXT_NOT_READY: missing frozen context",
            )
        return SimpleNamespace(
            status=ContextStatus.READY,
            context=SimpleNamespace(expected_absolute_return_15m=0.01),
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

    assert first.status is CandidateStatus.READY
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

    assert result.status is CandidateStatus.PRE_MOVE_NOT_READY
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

    assert result.status is CandidateStatus.PRE_CONTEXT_NOT_READY
    assert boundary.calls == []
