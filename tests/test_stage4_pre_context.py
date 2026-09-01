import asyncio
import inspect
import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from math import isclose
from pathlib import Path
from types import SimpleNamespace

import pytest

import stocker_execution.pre_context as pre_context_module
from stocker_core.config import IbkrConfig
from stocker_core.runs import Environment
from stocker_execution.history import IbkrHistoryCache
from stocker_execution.ibkr import (
    HistoricalBar,
    IbkrConnection,
    OptionChainDefinition,
    OptionContractRequest,
    OptionMarketSnapshot,
    QualifiedInstrument,
    QualifiedOption,
)
from stocker_execution.pre_context import (
    IBKR_MODEL_OPTION_COMPUTATION_TICK_TYPE,
    IBKR_MODEL_OPTION_IV_SOURCE,
    ContextStatus,
    OptionCandidate,
    PriorSessionContextService,
    PriorSessionContextStore,
    _contiguous_five_minute_ranges,
    _five_minute_starts,
    _session_contract,
    calculate_volatility,
    select_canonical_option_pair,
)


class SnapshotClient:
    def __init__(self, ticker: object) -> None:
        self.ticker = ticker
        self.connected = False
        self.generic_tick_lists: list[str] = []

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

    async def reqHistoricalDataAsync(self, contract: object, **kwargs: object) -> list[object]:
        return []

    async def reqTickersAsync(
        self, *contracts: object, regulatorySnapshot: bool = False
    ) -> list[object]:
        return [self.ticker]

    def reqMktData(
        self,
        contract: object,
        genericTickList: str = "",
        snapshot: bool = False,
        regulatorySnapshot: bool = False,
    ) -> object:
        self.generic_tick_lists.append(genericTickList)
        return self.ticker

    def cancelMktData(self, contract: object) -> bool:
        return True

    async def reqSecDefOptParamsAsync(
        self,
        underlyingSymbol: str,
        futFopExchange: str,
        underlyingSecType: str,
        underlyingConId: int,
    ) -> list[object]:
        return []


def option() -> QualifiedOption:
    return QualifiedOption(
        symbol="AMD",
        con_id=99001,
        exchange="SMART",
        currency="USD",
        security_type="OPT",
        expiry=date(2026, 9, 18),
        strike=100,
        right="C",
        multiplier="100",
        trading_class="AMD",
    )


def connected_snapshot_boundary(ticker: object) -> tuple[IbkrConnection, SnapshotClient]:
    client = SnapshotClient(ticker)
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=44,
            request_timeout_seconds=0.01,
        ),
        client=client,
    )
    asyncio.run(connection.connect())
    return connection, client


def test_canonical_volatility_arithmetic_uses_row_specific_model_iv() -> None:
    first = calculate_volatility(call_model_iv=0.24, put_model_iv=0.28)
    second = calculate_volatility(call_model_iv=0.36, put_model_iv=0.40)

    assert first.atm_iv == 0.26
    assert isclose(first.expected_absolute_return_15m, 0.0025628693370347896)
    assert second.atm_iv == 0.38
    assert second.expected_absolute_return_15m != first.expected_absolute_return_15m


def test_previous_session_grid_respects_known_xnys_half_day() -> None:
    observation, session_open, session_close, target_open = _session_contract(
        date(2025, 12, 1)
    )

    assert observation == date(2025, 11, 28)
    assert session_open == datetime(2025, 11, 28, 14, 30, tzinfo=UTC)
    assert session_close == datetime(2025, 11, 28, 18, 0, tzinfo=UTC)
    assert target_open == datetime(2025, 12, 1, 14, 30, tzinfo=UTC)
    assert len(_five_minute_starts(session_open, session_close)) == 42


def test_missing_history_is_grouped_into_only_contiguous_fetch_ranges() -> None:
    start = datetime(2026, 8, 31, 13, 30, tzinfo=UTC)
    missing = (start, start + timedelta(minutes=5), start + timedelta(minutes=20))

    groups = _contiguous_five_minute_ranges(missing)

    assert groups == (missing[:2], missing[2:])


def candidate(
    *,
    con_id: int,
    right: str,
    strike: float,
    expiry: date = date(2026, 9, 18),
    model_iv: float | None = 0.25,
    open_interest: float | None = 100,
    bid: float | None = 1.0,
    ask: float | None = 1.2,
) -> OptionCandidate:
    return OptionCandidate(
        con_id=con_id,
        symbol="AMD",
        exchange="SMART",
        currency="USD",
        expiry=expiry,
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        open_interest=open_interest,
        model_iv=model_iv,
        model_delta=0.5 if right == "C" else -0.5,
        model_gamma=0.02,
    )


def test_selector_preserves_frozen_expiry_strike_and_quality_tie_breaking() -> None:
    candidates = (
        candidate(con_id=31, right="C", strike=99, open_interest=100),
        candidate(con_id=32, right="P", strike=99, open_interest=100),
        candidate(con_id=21, right="C", strike=101, open_interest=200),
        candidate(con_id=22, right="P", strike=101, open_interest=200),
        candidate(
            con_id=11,
            right="C",
            strike=100,
            expiry=date(2026, 9, 25),
            open_interest=500,
        ),
        candidate(
            con_id=12,
            right="P",
            strike=100,
            expiry=date(2026, 9, 25),
            open_interest=500,
        ),
    )

    selected = select_canonical_option_pair(
        candidates,
        previous_close=100,
        observation_date=date(2026, 9, 1),
    )

    assert selected.call.con_id == 21
    assert selected.put.con_id == 22
    assert selected.expiry == date(2026, 9, 18)
    assert selected.strike == 101


def test_selector_fails_closed_when_ranked_pair_has_missing_model_iv() -> None:
    candidates = (
        candidate(con_id=11, right="C", strike=100, model_iv=None),
        candidate(con_id=12, right="P", strike=100),
        candidate(con_id=21, right="C", strike=101),
        candidate(con_id=22, right="P", strike=101),
    )

    with pytest.raises(ValueError, match="model IV"):
        select_canonical_option_pair(
            candidates,
            previous_close=100,
            observation_date=date(2026, 9, 1),
        )


def test_selector_rejects_invalid_selected_pair_without_substituting() -> None:
    candidates = (
        candidate(con_id=11, right="C", strike=100, open_interest=9),
        candidate(con_id=12, right="P", strike=100),
        candidate(con_id=21, right="C", strike=101),
        candidate(con_id=22, right="P", strike=101),
    )

    with pytest.raises(ValueError, match="open interest below 10"):
        select_canonical_option_pair(
            candidates,
            previous_close=100,
            observation_date=date(2026, 9, 1),
        )


def test_stage4_module_contains_no_stage5_threshold_or_calculation() -> None:
    source = inspect.getsource(pre_context_module)

    assert "0.475764059845861" not in source
    assert "PRE_MOVE_M" not in source


def test_stage2_option_snapshot_uses_only_tick13_model_computation_iv() -> None:
    ticker = SimpleNamespace(
        time=datetime(2026, 9, 1, 20, 1, tzinfo=UTC),
        bid=1.0,
        ask=1.2,
        callOpenInterest=120,
        putOpenInterest=None,
        modelGreeks=SimpleNamespace(impliedVol=0.27, delta=0.51, gamma=0.02),
        bidGreeks=SimpleNamespace(impliedVol=0.20),
        askGreeks=SimpleNamespace(impliedVol=0.34),
        lastGreeks=SimpleNamespace(impliedVol=0.99),
        impliedVolatility=0.88,
    )

    boundary, client = connected_snapshot_boundary(ticker)
    snapshots = asyncio.run(boundary.option_snapshots((option(),)))

    assert snapshots[0].model_iv == 0.27
    assert IBKR_MODEL_OPTION_COMPUTATION_TICK_TYPE == 13
    assert snapshots[0].model_delta == 0.51
    assert snapshots[0].model_gamma == 0.02
    assert not hasattr(snapshots[0], "bid_iv")
    assert not hasattr(snapshots[0], "generic_iv")
    assert client.generic_tick_lists == ["101"]
    assert "106" not in client.generic_tick_lists


def test_stage2_does_not_fallback_when_tick13_model_computation_is_missing() -> None:
    ticker = SimpleNamespace(
        time=datetime(2026, 9, 1, 20, 1, tzinfo=UTC),
        bid=1.0,
        ask=1.2,
        callOpenInterest=120,
        putOpenInterest=None,
        modelGreeks=None,
        bidGreeks=SimpleNamespace(impliedVol=0.20),
        askGreeks=SimpleNamespace(impliedVol=0.34),
        lastGreeks=SimpleNamespace(impliedVol=0.99),
        impliedVolatility=0.88,
    )

    boundary, client = connected_snapshot_boundary(ticker)
    snapshots = asyncio.run(boundary.option_snapshots((option(),)))

    assert snapshots[0].model_iv is None
    assert client.generic_tick_lists == ["101"]


class ContextBoundary(IbkrConnection):
    def __init__(self, *, missing_right: str | None = None) -> None:
        self.missing_right = missing_right
        self.history_requests = 0
        self.option_requests = 0

    async def historical_bars(
        self,
        instrument: QualifiedInstrument,
        **kwargs: object,
    ) -> tuple[HistoricalBar, ...]:
        self.history_requests += 1
        start = datetime(2026, 8, 31, 13, 30, tzinfo=UTC)
        return tuple(
            HistoricalBar(
                timestamp=start + timedelta(minutes=5 * index),
                open=100,
                high=101,
                low=99,
                close=100 + index / 100,
                volume=1_000,
            )
            for index in range(78)
        )

    async def option_chains(
        self, instrument: QualifiedInstrument
    ) -> tuple[OptionChainDefinition, ...]:
        self.option_requests += 1
        return (
            OptionChainDefinition(
                exchange="SMART",
                underlying_con_id=instrument.con_id,
                trading_class=instrument.symbol,
                multiplier="100",
                expirations=(date(2026, 9, 18),),
                strikes=(99, 100, 101),
            ),
        )

    async def qualify_options(
        self, requests: tuple[OptionContractRequest, ...]
    ) -> tuple[QualifiedOption, ...]:
        return tuple(
            QualifiedOption(
                symbol=item.symbol,
                con_id=99001 if item.right == "C" else 99002,
                exchange=item.exchange,
                currency=item.currency,
                security_type="OPT",
                expiry=item.expiry,
                strike=item.strike,
                right=item.right,
                multiplier=item.multiplier,
                trading_class=item.trading_class,
            )
            for item in requests
        )

    async def option_snapshots(
        self, options: tuple[QualifiedOption, ...]
    ) -> tuple[OptionMarketSnapshot, ...]:
        return tuple(
            OptionMarketSnapshot(
                option=item,
                captured_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
                bid=1.0,
                ask=1.2,
                open_interest=100,
                model_iv=None
                if item.right == self.missing_right
                else (0.24 if item.right == "C" else 0.28),
                model_delta=0.5 if item.right == "C" else -0.5,
                model_gamma=0.02,
            )
            for item in options
        )


def qualified_stock() -> QualifiedInstrument:
    return QualifiedInstrument(
        symbol="AMD",
        con_id=4391,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
    )


def test_service_persists_conid_lineage_and_reuses_context(tmp_path: Path) -> None:
    path = tmp_path / "stage4.sqlite3"
    boundary = ContextBoundary()
    service = PriorSessionContextService(
        boundary,
        IbkrHistoryCache(path),
        PriorSessionContextStore(path),
        clock=lambda: datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
    )

    first = asyncio.run(service.get_or_create(qualified_stock(), session=date(2026, 9, 1)))
    same_con_id_from_another_consumer = replace(qualified_stock(), symbol="AMD.QUALIFIED")
    second = asyncio.run(
        service.get_or_create(same_con_id_from_another_consumer, session=date(2026, 9, 1))
    )

    assert first.status is ContextStatus.READY
    assert first.context is not None
    assert first.context.underlying_con_id == 4391
    assert first.context.call_con_id == 99001
    assert first.context.put_con_id == 99002
    assert first.context.iv_source == IBKR_MODEL_OPTION_IV_SOURCE
    assert first.context.atm_iv == 0.26
    assert second.reused is True
    assert boundary.history_requests == 1
    assert boundary.option_requests == 1
    assert not hasattr(first.context, "p0")
    assert not hasattr(first.context, "m_price")
    assert not hasattr(first.context, "pre_move_m")


def test_invalid_persisted_context_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "stage4.sqlite3"
    service = PriorSessionContextService(
        ContextBoundary(),
        IbkrHistoryCache(path),
        PriorSessionContextStore(path),
        clock=lambda: datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
    )
    ready = asyncio.run(service.get_or_create(qualified_stock(), session=date(2026, 9, 1)))
    assert ready.status is ContextStatus.READY
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE ibkr_pre_volatility_contexts SET atm_iv = 4.2")

    result = asyncio.run(service.get_or_create(qualified_stock(), session=date(2026, 9, 1)))

    assert result.status is ContextStatus.NOT_READY
    assert result.context is None
    assert "does not match canonical arithmetic" in result.reason


@pytest.mark.parametrize("missing_right", ["C", "P"])
def test_service_returns_not_ready_for_missing_required_model_iv(
    tmp_path: Path, missing_right: str
) -> None:
    path = tmp_path / "stage4.sqlite3"
    service = PriorSessionContextService(
        ContextBoundary(missing_right=missing_right),
        IbkrHistoryCache(path),
        PriorSessionContextStore(path),
        clock=lambda: datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
    )

    result = asyncio.run(service.get_or_create(qualified_stock(), session=date(2026, 9, 1)))

    assert result.status is ContextStatus.NOT_READY
    assert result.context is None
    assert "model IV" in result.reason
