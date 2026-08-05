from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from stocker_prospective.ibkr import IBKRConnectionConfig, IBKRMarketDataAdapter
from stocker_prospective.market_data import MarketDataBudget, MarketDataType


def adapter_with(client: Any) -> IBKRMarketDataAdapter:
    adapter = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=4002,
            client_id=91,
            expected_environment="paper",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=1,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=MarketDataBudget(
            line_limit=20,
            reserved_headroom=1,
            request_rate_limit=100,
        ),
    )
    adapter._client = client
    return adapter


class HighResolutionClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, tuple[Any, ...]]] = []
        self.cancellations: list[tuple[str, tuple[Any, ...]]] = []

    def reqTickByTickData(self, *arguments: Any) -> None:  # noqa: N802
        self.requests.append(("tick_by_tick", arguments))

    def cancelTickByTickData(self, *arguments: Any) -> None:  # noqa: N802
        self.cancellations.append(("tick_by_tick", arguments))

    def reqMktDepth(self, *arguments: Any) -> None:  # noqa: N802
        self.requests.append(("depth", arguments))

    def cancelMktDepth(self, *arguments: Any) -> None:  # noqa: N802
        self.cancellations.append(("depth", arguments))

    def reqHistoricalData(self, *arguments: Any) -> None:  # noqa: N802
        self.requests.append(("historical", arguments))

    def cancelHistoricalData(self, *arguments: Any) -> None:  # noqa: N802
        self.cancellations.append(("historical", arguments))

    def cancelMktData(self, request_id: int) -> None:  # noqa: N802
        raise AssertionError(f"incorrect cancellation surface for {request_id}")

    def reqCurrentTime(self) -> None:  # noqa: N802
        self.requests.append(("current_time", ()))

    def reqMktDepthExchanges(self) -> None:  # noqa: N802
        self.requests.append(("depth_exchanges", ()))

    def reqMarketDataType(self, market_data_type: int) -> None:  # noqa: N802
        self.requests.append(("market_data_type", (market_data_type,)))

    def serverVersion(self) -> int:  # noqa: N802
        return 187


def test_tick_by_tick_bidask_and_last_are_bounded_and_cancel_exactly() -> None:
    client = HighResolutionClient()
    adapter = adapter_with(client)
    contract = object()

    bidask_id = adapter.request_tick_by_tick(
        contract,
        subscription_key="AAL:bidask",
        tick_type="BidAsk",
    )
    last_id = adapter.request_tick_by_tick(
        contract,
        subscription_key="AAL:last",
        tick_type="Last",
    )

    assert client.requests == [
        ("tick_by_tick", (bidask_id, contract, "BidAsk", 0, False)),
        ("tick_by_tick", (last_id, contract, "Last", 0, False)),
    ]

    adapter.cancel_tick_by_tick(bidask_id, subscription_key="AAL:bidask")
    adapter.cancel_tick_by_tick(last_id, subscription_key="AAL:last")
    assert client.cancellations == [
        ("tick_by_tick", (bidask_id,)),
        ("tick_by_tick", (last_id,)),
    ]


def test_depth_and_five_minute_keep_up_to_date_use_official_request_contract() -> None:
    client = HighResolutionClient()
    adapter = adapter_with(client)
    contract = object()

    depth_id = adapter.request_market_depth(
        contract,
        subscription_key="AAL:depth",
        rows=5,
        smart_depth=True,
    )
    bar_id = adapter.request_historical_five_minute_updates(
        contract,
        subscription_key="AAL:five_minute",
    )

    assert client.requests[0] == ("depth", (depth_id, contract, 5, True, []))
    assert client.requests[1] == (
        "historical",
        (
            bar_id,
            contract,
            "",
            "1 D",
            "5 mins",
            "TRADES",
            1,
            2,
            True,
            [],
        ),
    )

    adapter.cancel_market_depth(depth_id, subscription_key="AAL:depth")
    adapter.cancel_historical_updates(bar_id, subscription_key="AAL:five_minute")
    assert client.cancellations == [
        ("depth", (depth_id, True)),
        ("historical", (bar_id,)),
    ]


def test_high_resolution_callbacks_preserve_order_and_depth_reset() -> None:
    adapter = adapter_with(HighResolutionClient())
    observed_at = datetime(2026, 7, 24, 14, 30, tzinfo=UTC)

    adapter.on_tick_by_tick_bidask(
        7,
        {
            "provider_timestamp_utc": observed_at.isoformat(),
            "bid": 12.0,
            "ask": 12.02,
        },
    )
    adapter.on_tick_by_tick_trade(
        8,
        {
            "provider_timestamp_utc": observed_at.isoformat(),
            "price": 12.02,
            "size": 100.0,
        },
    )
    adapter.on_depth_update(9, {"operation": "update", "side": "bid", "position": 0})
    adapter.on_depth_reset(9, "ibkr_market_depth_reset")

    events = adapter.drain_stream_events()
    assert [event["kind"] for event in events] == [
        "tick_by_tick_bidask",
        "tick_by_tick_trade",
        "depth",
        "depth_reset",
    ]
    assert [event["source_sequence"] for event in events] == [1, 2, 3, 4]
    assert all(event["received_monotonic_ns"] > 0 for event in events)


def test_capability_requests_are_market_data_only() -> None:
    client = HighResolutionClient()
    adapter = adapter_with(client)

    adapter.require_live_market_data()
    adapter.request_current_time()
    adapter.request_depth_exchanges()

    assert adapter.server_version() == 187
    assert client.requests == [
        ("market_data_type", (1,)),
        ("current_time", ()),
        ("depth_exchanges", ()),
    ]


def test_current_time_callback_preserves_its_request_boundary() -> None:
    adapter = adapter_with(HighResolutionClient())
    provider_at = datetime.now(UTC).replace(microsecond=0)

    adapter.request_current_time()
    adapter.on_current_time(provider_at)

    event = adapter.drain_stream_events()[0]
    assert event["kind"] == "current_time"
    assert event["provider_timestamp_utc"] == provider_at.isoformat()
    assert datetime.fromisoformat(event["clock_probe_requested_at_utc"]) <= datetime.fromisoformat(
        event["received_timestamp_utc"]
    )
    assert event["clock_probe_requested_monotonic_ns"] <= event["received_monotonic_ns"]


def test_disconnected_server_version_is_absent_instead_of_crashing() -> None:
    class DisconnectedClient:
        def serverVersion(self) -> None:  # noqa: N802
            return None

    adapter = adapter_with(DisconnectedClient())

    assert adapter.server_version() is None
