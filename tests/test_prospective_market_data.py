from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from stocker_prospective.market_data import (
    BoundedCallbackRegistry,
    BoundedRealtimeBarQueue,
    BoundedStreamQuoteCache,
    CallbackRequestError,
    ConnectionEventKind,
    ConnectionState,
    ConnectionTracker,
    MarketDataBudget,
    MarketDataBudgetError,
    MarketDataType,
    RealtimeBarUpdate,
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


def test_metadata_requests_consume_rate_budget_without_market_data_lines() -> None:
    budget = MarketDataBudget(
        line_limit=20,
        reserved_headroom=2,
        request_rate_limit=2,
        request_rate_window_seconds=1,
    )

    budget.reserve("metadata-a", lines=0, now=10.0)
    budget.reserve("metadata-b", lines=0, now=10.1)

    assert budget.snapshot(now=10.2).active_lines == 0
    assert budget.snapshot(now=10.2).pending_requests == 2
    with pytest.raises(MarketDataBudgetError, match="request_rate"):
        budget.reserve("metadata-c", lines=0, now=10.2)


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


@pytest.mark.parametrize("code", [2104, 2106, 2107, 2108, 2158])
def test_ibkr_connection_notifications_do_not_degrade_or_fail_requests(code: int) -> None:
    from stocker_prospective.ibkr import IBKRConnectionConfig, IBKRMarketDataAdapter

    adapter = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=7497,
            client_id=71,
            expected_environment="paper",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=15,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=MarketDataBudget(
            line_limit=4,
            reserved_headroom=1,
            request_rate_limit=20,
        ),
    )
    adapter.on_connected(MarketDataType.LIVE)
    adapter.callbacks.begin(17, kind="quote")

    adapter.on_error(17, code, "official connectivity notification")

    health = adapter.connection.health()
    assert health.state is ConnectionState.CONNECTED
    assert health.last_error_code is None
    assert health.last_message == "connected"
    assert adapter.callbacks.is_pending(17)
    assert adapter.connection.events[-1].code == code
    assert adapter.connection.events[-1].message == "official connectivity notification"
    assert adapter.connection.events[-1].state is ConnectionState.CONNECTED
    assert (
        adapter.connection.events[-1].event_kind
        is ConnectionEventKind.INFORMATIONAL_NOTIFICATION
    )


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


def test_adapter_rejects_inherited_order_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stocker_prospective.ibkr as ibkr_module
    from stocker_prospective.ibkr import IBKRConnectionConfig, IBKRMarketDataAdapter

    monkeypatch.setattr(ibkr_module, "require_official_ibkr_api", lambda: object())
    adapter = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=7497,
            client_id=71,
            expected_environment="paper",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=15,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=MarketDataBudget(
            line_limit=4,
            reserved_headroom=1,
            request_rate_limit=20,
        ),
    )

    class OrderCapableBase:
        def placeOrder(self) -> None:  # noqa: N802
            return None

    class UnsafeInheritedClient(OrderCapableBase):
        def connect(self) -> bool:
            return True

        def disconnect(self) -> None:
            return None

        def run(self) -> None:
            return None

        def reqMktData(self) -> None:  # noqa: N802
            return None

        def cancelMktData(self) -> None:  # noqa: N802
            return None

    with pytest.raises(TypeError, match="order-capable"):
        adapter.attach_official_client(UnsafeInheritedClient())


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


def test_callback_registry_evicts_finished_results_at_a_fixed_bound() -> None:
    registry = BoundedCallbackRegistry(
        max_pending_requests=2,
        max_items_per_request=2,
        max_finished_requests=1,
    )
    registry.begin(1, kind="first")
    registry.complete(1)
    registry.wait(1, timeout_seconds=0)
    registry.begin(2, kind="second")
    registry.complete(2)
    registry.wait(2, timeout_seconds=0)

    # Request 1 has been evicted, so its ID may safely be reused.
    registry.begin(1, kind="reused")


def test_stream_quote_cache_is_bounded_and_keeps_only_latest_fields() -> None:
    cache = BoundedStreamQuoteCache(max_subscriptions=1, max_fields_per_subscription=2)
    cache.register(10)
    cache.add(10, {"field": "bid", "value": 1.0})
    cache.add(10, {"field": "bid", "value": 1.1})
    cache.add(10, {"field": "ask", "value": 1.2})

    assert cache.snapshot(10) == (
        {"field": "bid", "value": 1.1},
        {"field": "ask", "value": 1.2},
    )
    with pytest.raises(CallbackRequestError, match="field_cache_exhausted"):
        cache.add(10, {"field": "last", "value": 1.15})
    with pytest.raises(CallbackRequestError, match="subscription_cache_exhausted"):
        cache.register(11)


def test_realtime_bar_queue_rejects_overflow_and_drains_in_order() -> None:
    queue = BoundedRealtimeBarQueue(max_items=1)
    update = RealtimeBarUpdate(
        request_id=1,
        source_timestamp_utc=datetime(2026, 7, 24, 14, 30, tzinfo=UTC),
        receive_timestamp_utc=datetime(2026, 7, 24, 14, 30, 1, tzinfo=UTC),
        open=12.0,
        high=12.1,
        low=11.9,
        close=12.05,
        volume=None,
        wap=None,
        trade_count=None,
    )
    queue.add(update)

    with pytest.raises(CallbackRequestError, match="realtime_bar_queue_exhausted"):
        queue.add(update)
    assert queue.drain() == (update,)
    assert queue.size == 0


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
        snapshot_flags: list[bool] = []

        def reqMktData(self, request_id: int, *arguments: object) -> None:
            self.snapshot_flags.append(bool(arguments[2]))
            adapter.on_quote_update(
                request_id,
                {"bid": 1.25, "ask": None, "market_data_type": "live"},
            )
            adapter.callbacks.complete(request_id)

        def cancelMktData(self, request_id: int) -> None:
            self.cancelled.append(request_id)

    client = FakeClient()
    adapter._client = client

    result = adapter.capture_temporary_quote(contract=object(), timeout_seconds=0)

    assert result.items == ({"bid": 1.25, "ask": None, "market_data_type": "live"},)
    assert client.cancelled == [result.request_id]
    assert client.snapshot_flags == [True]
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


def test_continuous_market_data_updates_use_bounded_stream_cache() -> None:
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
            quote_capture_timeout_seconds=15,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=budget,
    )

    class StreamingClient:
        def reqMktData(self, request_id: int, *_: object) -> None:
            adapter.on_quote_update(request_id, {"field": "bid", "value": 1.0})
            adapter.on_quote_update(request_id, {"field": "bid", "value": 1.1})

        def cancelMktData(self, request_id: int) -> None:
            return None

    adapter._client = StreamingClient()
    request_id = adapter.request_market_data(object(), subscription_key="AAL")

    assert adapter.stream_quotes.snapshot(request_id) == ({"field": "bid", "value": 1.1},)
    assert budget.snapshot().active_lines == 1
    adapter.cancel_market_data(request_id, subscription_key="AAL")
    assert budget.snapshot().active_lines == 0


def test_realtime_bar_subscription_uses_correct_cancel_and_lost_data_cleanup() -> None:
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
            quote_capture_timeout_seconds=15,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=budget,
    )

    class RealtimeClient:
        cancelled: list[int] = []

        def reqRealTimeBars(self, request_id: int, *_: object) -> None:
            adapter.on_realtime_bar(
                RealtimeBarUpdate(
                    request_id=request_id,
                    source_timestamp_utc=datetime(2026, 7, 24, 14, 30, tzinfo=UTC),
                    receive_timestamp_utc=datetime(2026, 7, 24, 14, 30, 1, tzinfo=UTC),
                    open=12.0,
                    high=12.1,
                    low=11.9,
                    close=12.05,
                    volume=10.0,
                    wap=12.03,
                    trade_count=2,
                )
            )

        def cancelRealTimeBars(self, request_id: int) -> None:
            self.cancelled.append(request_id)

        def cancelMktData(self, request_id: int) -> None:
            raise AssertionError(f"wrong cancellation method for {request_id}")

    client = RealtimeClient()
    adapter._client = client
    request_id = adapter.request_realtime_bars(object(), subscription_key="AAL-bars")

    assert adapter.realtime_bars.drain()[0].request_id == request_id
    adapter.on_error(-1, 1101, "data lost")
    assert budget.snapshot().active_lines == 0
    assert adapter.subscriptions.active_count == 0
    assert adapter.connection.health().subscriptions_require_rebuild is True
    assert client.cancelled == []

    rebuilt_id = adapter.request_realtime_bars(object(), subscription_key="AAL-bars")
    adapter.connection.subscriptions_rebuilt()
    adapter.cancel_realtime_bars(rebuilt_id, subscription_key="AAL-bars")
    assert client.cancelled == [rebuilt_id]
    assert adapter.connection.health().subscriptions_require_rebuild is False


def test_option_metadata_and_qualification_are_rate_bounded() -> None:
    from stocker_prospective.ibkr import IBKRConnectionConfig, IBKRMarketDataAdapter

    budget = MarketDataBudget(
        line_limit=10,
        reserved_headroom=1,
        request_rate_limit=1,
    )
    adapter = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=7497,
            client_id=71,
            expected_environment="paper",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=15,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=budget,
    )

    class MetadataClient:
        def reqSecDefOptParams(self, request_id: int, *_: object) -> None:
            adapter.on_option_parameter_end(request_id)

        def reqContractDetails(self, request_id: int, contract: object) -> None:
            adapter.on_contract_details(request_id, contract)
            adapter.on_contract_details_end(request_id)

    adapter._client = MetadataClient()
    adapter.request_option_chain_metadata(
        underlying_symbol="AAL",
        exchange="",
        underlying_security_type="STK",
        underlying_contract_id=1,
    )

    with pytest.raises(MarketDataBudgetError, match="request_rate"):
        adapter.qualify_exact_contract(object())

    assert budget.snapshot().active_lines == 0
