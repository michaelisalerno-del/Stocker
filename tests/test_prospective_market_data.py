from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from stocker_prospective.market_data import (
    BoundedCallbackRegistry,
    CallbackRequestError,
    ConnectionState,
    ConnectionTracker,
    MarketDataBudget,
    MarketDataBudgetError,
    MarketDataType,
    RequestIdAllocator,
    SubscriptionRegistry,
    classify_ibkr_error,
)
from stocker_prospective.options import (
    DteBucket,
    bounded_contract_requests,
    select_atm_strike,
    select_expiries,
)


def test_request_ids_are_monotonic_and_safe_after_server_sync() -> None:
    allocator = RequestIdAllocator(start=7)

    assert allocator.next() == 7
    allocator.synchronise(100)
    assert allocator.next() == 100
    allocator.synchronise(10)
    assert allocator.next() == 101


def test_market_data_budget_reserves_capacity_and_deduplicates() -> None:
    budget = MarketDataBudget(
        line_limit=10,
        reserved_headroom=2,
        request_rate_limit=100,
        max_waiting_signals=1,
    )

    for number in range(8):
        budget.reserve(f"option-{number}", lines=1, now=number / 100)

    budget.reserve("option-0", lines=1, now=0.5)
    assert budget.snapshot().active_lines == 8

    with pytest.raises(MarketDataBudgetError, match="blocked_market_data_budget_exhausted"):
        budget.reserve("one-too-many", lines=1, now=0.6)

    assert budget.snapshot().rejected_signals == 1
    budget.request_cancellation("option-0")
    assert budget.snapshot().awaiting_cancellation == 1
    budget.confirm_cancellation("option-0")
    assert budget.snapshot().active_lines == 7


def test_market_data_budget_enforces_request_rate() -> None:
    budget = MarketDataBudget(
        line_limit=20,
        reserved_headroom=2,
        request_rate_limit=2,
        request_rate_window_seconds=1,
    )
    budget.reserve("a", now=10.0)
    budget.reserve("b", now=10.1)

    with pytest.raises(MarketDataBudgetError, match="request_rate"):
        budget.reserve("c", now=10.2)

    budget.reserve("c", now=11.1)
    assert budget.snapshot().active_lines == 3


def test_connection_tracker_distinguishes_maintained_and_lost_data_reconnects() -> None:
    tracker = ConnectionTracker()
    tracker.connected(MarketDataType.LIVE)
    tracker.connection_lost(code=1100, message="Connectivity lost")
    assert tracker.health().state is ConnectionState.DISCONNECTED

    tracker.connection_restored(data_maintained=True, code=1102)
    assert tracker.health().state is ConnectionState.CONNECTED
    assert tracker.health().subscriptions_require_rebuild is False

    tracker.connection_lost(code=1100, message="Connectivity lost")
    tracker.connection_restored(data_maintained=False, code=1101)
    assert tracker.health().subscriptions_require_rebuild is True

    tracker.socket_port_reset(7497)
    assert tracker.health().state is ConnectionState.PORT_RESET
    assert tracker.health().subscriptions_require_rebuild is True
    assert [event.code for event in tracker.events] == [None, 1100, 1102, 1100, 1101, 1300]


def test_shutdown_clears_pending_requests_without_inventing_callbacks() -> None:
    budget = MarketDataBudget(
        line_limit=5,
        reserved_headroom=1,
        request_rate_limit=100,
    )
    budget.reserve("pending-a")
    budget.reserve("pending-b")

    cancelled = budget.shutdown()

    assert cancelled == ("pending-a", "pending-b")
    assert budget.snapshot().active_lines == 0
    assert budget.snapshot().pending_requests == 0


def test_expiry_buckets_never_substitute_outside_their_calendar_day_range() -> None:
    session = date(2026, 7, 24)
    selected = select_expiries(
        session,
        [
            date(2026, 7, 24),
            date(2026, 7, 25),
            date(2026, 7, 26),
            date(2026, 7, 27),
            date(2026, 7, 29),
        ],
    )

    assert selected[DteBucket.ZERO_DTE].expiry == date(2026, 7, 24)
    assert selected[DteBucket.ONE_DTE].expiry == date(2026, 7, 25)
    assert selected[DteBucket.THREE_TO_FIVE_DTE].expiry == date(2026, 7, 27)

    missing = select_expiries(session, [date(2026, 7, 26), date(2026, 7, 30)])
    assert missing[DteBucket.ZERO_DTE].reason == "no_expiry_in_bucket"
    assert missing[DteBucket.ONE_DTE].reason == "no_expiry_in_bucket"
    assert missing[DteBucket.THREE_TO_FIVE_DTE].reason == "no_expiry_in_bucket"


def test_atm_tie_uses_lower_strike_and_surface_is_bounded() -> None:
    strikes = [95.0, 100.0, 105.0, 110.0, 115.0]
    assert select_atm_strike(102.5, strikes) == 100.0

    requests = bounded_contract_requests(
        underlying_contract_id=123,
        expiry=date(2026, 7, 27),
        strikes=strikes,
        underlying_reference=102.5,
        strike_steps=1,
        exchange="SMART",
        trading_class="XYZ",
    )

    assert {(item.strike, item.right) for item in requests} == {
        (95.0, "C"),
        (95.0, "P"),
        (100.0, "C"),
        (100.0, "P"),
        (105.0, "C"),
        (105.0, "P"),
    }
    assert all(item.exact_qualification_required for item in requests)


def test_market_data_adapter_contract_has_no_order_surface() -> None:
    from stocker_prospective.ibkr import IBKRMarketDataAdapter

    public_names = {name for name in dir(IBKRMarketDataAdapter) if not name.startswith("_")}

    assert "place_order" not in public_names
    assert "cancel_order" not in public_names
    assert "submit_order" not in public_names


def test_market_data_types_keep_missing_values_and_primary_eligibility_distinct() -> None:
    assert MarketDataType.LIVE.primary_eligible is True
    assert MarketDataType.FROZEN.primary_eligible is False
    assert MarketDataType.DELAYED.primary_eligible is False
    assert MarketDataType.DELAYED_FROZEN.primary_eligible is False

    now = datetime.now(UTC)
    assert now - timedelta(minutes=1) < now


def test_callback_registry_correlates_bounds_times_out_and_preserves_none() -> None:
    registry = BoundedCallbackRegistry(max_pending_requests=2, max_items_per_request=2)
    registry.begin(11, kind="quote")
    registry.add(11, {"bid": 1.0, "ask": None})
    registry.complete(11)

    result = registry.wait(11, timeout_seconds=0)

    assert result.complete is True
    assert result.items == ({"bid": 1.0, "ask": None},)

    registry.begin(12, kind="contract")
    with pytest.raises(CallbackRequestError, match="incomplete_callback_timeout"):
        registry.wait(12, timeout_seconds=0)

    registry.begin(13, kind="chain")
    registry.add(13, "first")
    registry.add(13, "second")
    with pytest.raises(CallbackRequestError, match="bounded_callback_queue_exhausted"):
        registry.add(13, "third")


def test_callback_shutdown_fails_pending_requests() -> None:
    registry = BoundedCallbackRegistry(max_pending_requests=2, max_items_per_request=2)
    registry.begin(20, kind="quote")
    registry.begin(21, kind="contract")

    assert registry.shutdown() == (20, 21)
    with pytest.raises(CallbackRequestError, match="shutdown_during_pending_request"):
        registry.wait(20, timeout_seconds=0)


def test_subscription_registry_avoids_duplicates_and_rebuilds_only_after_data_loss() -> None:
    subscriptions = SubscriptionRegistry()

    assert subscriptions.register("AAL-underlying", 10) is True
    assert subscriptions.register("AAL-underlying", 99) is False
    assert subscriptions.after_reconnect(data_maintained=True) == ()
    assert subscriptions.after_reconnect(data_maintained=False) == ("AAL-underlying",)
    assert subscriptions.register("AAL-underlying", 12) is True
    assert subscriptions.active_count == 1
    assert subscriptions.remove("AAL-underlying") == 12
    assert subscriptions.active_count == 0


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (100, "blocked_ibkr_market_data_subscription:pacing_error"),
        (101, "blocked_market_data_budget_exhausted"),
        (354, "blocked_ibkr_market_data_subscription:missing_subscription"),
        (10197, "blocked_ibkr_market_data_subscription:competing_session"),
        (9999, "blocked_ibkr_market_data_subscription:ibkr_error_9999"),
    ],
)
def test_ibkr_errors_are_actionable(code: int, expected: str) -> None:
    assert classify_ibkr_error(code) == expected


def test_temporary_quote_capture_is_cancelled_and_missing_values_remain_none() -> None:
    from stocker_prospective.ibkr import IBKRConnectionConfig, IBKRMarketDataAdapter

    budget = MarketDataBudget(
        line_limit=4,
        reserved_headroom=1,
        request_rate_limit=20,
    )
    adapter = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=7497,
            client_id=71,
            expected_environment="paper",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=1,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=budget,
    )

    class FakeClient:
        cancelled: list[int] = []

        def reqMktData(self, request_id: int, *_: object) -> None:
            adapter.on_quote_update(
                request_id,
                {"bid": 1.25, "ask": None, "market_data_type": "live"},
                complete=True,
            )

        def cancelMktData(self, request_id: int) -> None:
            self.cancelled.append(request_id)

    client = FakeClient()
    adapter._client = client

    result = adapter.capture_temporary_quote(contract=object(), timeout_seconds=0)

    assert result.items == ({"bid": 1.25, "ask": None, "market_data_type": "live"},)
    assert client.cancelled == [result.request_id]
    assert budget.snapshot().active_lines == 0


def test_incomplete_temporary_capture_times_out_and_is_still_cancelled() -> None:
    from stocker_prospective.ibkr import IBKRConnectionConfig, IBKRMarketDataAdapter

    budget = MarketDataBudget(
        line_limit=4,
        reserved_headroom=1,
        request_rate_limit=20,
    )
    adapter = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=7497,
            client_id=71,
            expected_environment="paper",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=1,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=budget,
    )

    class SilentClient:
        cancelled: list[int] = []

        def reqMktData(self, request_id: int, *_: object) -> None:
            return None

        def cancelMktData(self, request_id: int) -> None:
            self.cancelled.append(request_id)

    client = SilentClient()
    adapter._client = client
    with pytest.raises(CallbackRequestError, match="incomplete_callback_timeout"):
        adapter.capture_temporary_quote(contract=object(), timeout_seconds=0)

    assert len(client.cancelled) == 1
    assert budget.snapshot().active_lines == 0
