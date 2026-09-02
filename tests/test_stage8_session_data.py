from datetime import UTC, date, datetime, time, timedelta

from stocker_core.runs import Environment, RunConfig, RunWindow
from stocker_execution.ibkr import HistoricalBar
from stocker_execution.runtime import (
    ExchangeSessionResolver,
    MarketSessionState,
)
from stocker_execution.stage5 import calculate_session_hard_inputs


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


def test_exchange_session_resolver_uses_run_timezone_and_closed_day() -> None:
    resolver = ExchangeSessionResolver()
    run = RunConfig(
        run_id="london",
        universe="FTSE",
        strategy="SESSION_HARD",
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
        strategy="SESSION_HARD",
        environment=Environment.PAPER,
    )

    try:
        resolver.resolve(run, datetime(2026, 9, 2, 8, 30, tzinfo=UTC))
    except ValueError as exc:
        assert "explicit market session" in str(exc)
    else:
        raise AssertionError("missing run session silently received a market fallback")
