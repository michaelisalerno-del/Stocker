import asyncio
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from stocker_core.runs import Environment, RunConfig, RunWindow
from stocker_execution.history import IbkrHistoryCache
from stocker_execution.ibkr import HistoricalBar, IbkrConnection, QualifiedInstrument
from stocker_execution.runtime import (
    ExchangeSessionResolver,
    IbkrSessionDataSource,
    MarketSessionState,
)
from stocker_execution.session_hard_structure_d import StrategyOpportunityKey
from stocker_execution.stage5 import (
    Stage5FeatureSnapshot,
    Stage5Status,
    calculate_session_hard_inputs,
)


def _bar(index: int) -> HistoricalBar:
    opening = 100.0 + index
    return HistoricalBar(
        timestamp=datetime(2026, 9, 2, 13, 30, tzinfo=UTC) + timedelta(minutes=5 * index),
        open=opening,
        high=opening + 1.0,
        low=opening - 1.0,
        close=opening + 0.5,
        volume=100.0 * (index + 1),
    )


def test_session_hard_inputs_use_only_exact_completed_checkpoint_prefix() -> None:
    features = calculate_session_hard_inputs(
        tuple(_bar(index) for index in range(6)),
        checkpoint=6,
        session_open=datetime(2026, 9, 2, 13, 30, tzinfo=UTC),
    )

    assert set(features) == {
        "range_effort",
        "travel_effort",
        "absolute_efficiency",
        "close_retention",
        "directional_persistence",
        "prior_6_mean_range",
        "prior_6_price_travel",
        "prior_6_absolute_net_movement",
        "recent_vs_earlier_range_ratio",
        "current_bar_range_vs_prior_6",
        "current_bar_body_fraction",
        "current_bar_extreme_wick_fraction",
        "current_volume_vs_session_mean",
        "current_volume_vs_prior6_mean",
        "last3_vs_first3_volume",
    }
    assert features["directional_persistence"] == 1.0
    assert features["current_bar_body_fraction"] == 0.25
    assert features["last3_vs_first3_volume"] == 2.5


def test_session_hard_inputs_reject_a_gap_or_future_bar() -> None:
    bars = tuple(_bar(index) for index in range(6))
    future = (
        *bars[:5],
        HistoricalBar(
            timestamp=datetime(2026, 9, 2, 14, 0, tzinfo=UTC),
            open=bars[5].open,
            high=bars[5].high,
            low=bars[5].low,
            close=bars[5].close,
            volume=bars[5].volume,
        ),
    )

    try:
        calculate_session_hard_inputs(
            future,
            checkpoint=6,
            session_open=datetime(2026, 9, 2, 13, 30, tzinfo=UTC),
        )
    except ValueError as exc:
        assert "exact completed" in str(exc)
    else:
        raise AssertionError("mismatched bar timestamp was accepted")


class SessionHistoryBoundary(IbkrConnection):
    def __init__(self) -> None:
        self.durations: list[str] = []

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
        del requested, end_time, minimum_bars
        assert bar_size == "5 mins"
        assert what_to_show == "TRADES"
        assert regular_trading_hours is True
        self.durations.append(duration)
        seconds = int(duration.removesuffix(" S"))
        covered_bars = max(1, (seconds + 299) // 300)
        return tuple(_bar(index) for index in range(6))[-covered_bars:]


def test_session_hard_context_requests_full_checkpoint_history_prefix(tmp_path: Path) -> None:
    boundary = SessionHistoryBoundary()
    source = IbkrSessionDataSource(
        boundary,
        IbkrHistoryCache(tmp_path / "history.sqlite3"),
    )
    run = RunConfig(
        run_id="us-hard-hv",
        universe="US_ALL_MID_CAP_BUCKETS_V1",
        strategy="SESSION_HARD_HV",
        environment=Environment.PAPER,
        session=RunWindow(
            start=time(9, 30),
            end=time(16),
            timezone="America/New_York",
            calendar="XNYS",
        ),
    )
    instrument = QualifiedInstrument("TEST", 123, "SMART", "NASDAQ", "USD", "STK")
    t0 = datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
    row = Stage5FeatureSnapshot(
        run_ids=(run.run_id,),
        universe_id=run.universe,
        con_id=instrument.con_id,
        symbol=instrument.symbol,
        session=t0.date(),
        t0=t0,
        status=Stage5Status.READY,
        exclusion_reason="",
        p0=100.0,
        expected_absolute_return_15m=0.01,
        m_price=1.0,
        raw_open_t0_minus_3m=99.0,
        raw_open_t0=100.0,
        alignment_factor=1.0,
        aligned_pre_open=99.0,
        raw_pre_move_price=1.0,
        pre_move_m=1.0,
        calculation_version="STAGE5_PRE_MOVE_HV_V1",
    )

    context = asyncio.run(
        source.context_for(
            run,
            (row,),
            6,
            {instrument.con_id: instrument},
            (),
        )
    )

    assert boundary.durations == ["2100 S"]
    assert StrategyOpportunityKey(instrument.con_id, row.session, t0) in context.session_hard


def test_exchange_session_resolver_uses_run_timezone_and_closed_day() -> None:
    resolver = ExchangeSessionResolver()
    run = RunConfig(
        run_id="london",
        universe="FTSE",
        strategy="SESSION_HARD_HV",
        environment=Environment.PAPER,
        session=RunWindow(
            start=time(8),
            end=time(16, 30),
            timezone="Europe/London",
            calendar="XLON",
        ),
    )

    active = resolver.resolve(run, datetime(2026, 9, 2, 8, 30, tzinfo=UTC))
    closed = resolver.resolve(run, datetime(2026, 9, 5, 8, 30, tzinfo=UTC))

    assert active.session == date(2026, 9, 2)
    assert active.state is MarketSessionState.ACTIVE_SESSION
    assert active.opens_at == datetime(2026, 9, 2, 7, 0, tzinfo=UTC)
    assert closed.state is MarketSessionState.CLOSED_DAY
    assert closed.checkpoint_times() == ()


def test_exchange_session_resolver_rejects_missing_session_instead_of_assuming_us() -> None:
    resolver = ExchangeSessionResolver()
    run = RunConfig(
        run_id="missing-session",
        universe="FTSE",
        strategy="SESSION_HARD_HV",
        environment=Environment.PAPER,
    )

    try:
        resolver.resolve(run, datetime(2026, 9, 2, 8, 30, tzinfo=UTC))
    except ValueError as exc:
        assert "explicit market session" in str(exc)
    else:
        raise AssertionError("missing run session silently received a market fallback")
