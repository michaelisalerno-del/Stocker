from __future__ import annotations

import asyncio
import inspect
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stocker_execution.broker import Broker
from stocker_execution.ibkr_observability.config import IBKRObserverConfig
from stocker_execution.ibkr_observability.fake import FakeIBKRObservabilityClient
from stocker_execution.ibkr_observability.ledger import (
    QuoteLedgerError,
    append_quote_observation,
)
from stocker_execution.ibkr_observability.models import (
    ContractIdentity,
    MarketDataType,
    ObservationClassification,
    ObservationPlanItem,
    QuoteObservationRecord,
    QuoteSnapshot,
)
from stocker_execution.ibkr_observability.observer import IBKRObserver, classify_snapshot
from stocker_execution.ibkr_observability.official_api import official_api_status
from stocker_execution.ibkr_observability.plan import build_observation_plan
from stocker_execution.ibkr_observability.protocol import IBKRObservabilityClient


def _contract() -> ContractIdentity:
    return ContractIdentity(
        research_symbol="AAPL",
        source_provider_symbol="AAPL.US",
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type="STK",
        currency="USD",
        routing_exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class="NMS",
        valid_exchanges="SMART,NASDAQ",
        minimum_tick=0.01,
        timezone_identifier="US/Eastern",
        trading_hours="20250707:0930-20250707:1600",
        liquid_hours="20250707:0930-20250707:1600",
        resolution_timestamp=datetime(2025, 7, 7, 13, 0, tzinfo=UTC),
        api_tws_version="test",
        resolution_status="resolved",
        resolution_error=None,
    )


def _snapshot(
    *,
    bid: float | None = 100.0,
    ask: float | None = 100.1,
    market_data_type: MarketDataType = MarketDataType.LIVE,
    response_delay: float = 1.0,
    completion_delay: float | None = None,
    snapshot_complete: bool = True,
) -> QuoteSnapshot:
    requested = datetime(2025, 7, 7, 14, 5, tzinfo=UTC)
    completed_after = response_delay + 0.1 if completion_delay is None else completion_delay
    return QuoteSnapshot(
        request_id=7,
        requested_timestamp=requested,
        server_time_observation=requested,
        local_send_timestamp=requested,
        first_response_timestamp=requested + timedelta(seconds=response_delay),
        snapshot_completion_timestamp=requested + timedelta(seconds=completed_after),
        bid=bid,
        ask=ask,
        bid_size=100.0 if bid is not None else None,
        ask_size=200.0 if ask is not None else None,
        last=100.05,
        last_size=50.0,
        market_data_type=market_data_type,
        snapshot_complete=snapshot_complete,
        error_code=None,
        error_message=None,
        connection_status="connected",
    )


def test_default_configuration_is_disabled_and_localhost_only() -> None:
    config = IBKRObserverConfig()

    assert not config.enabled
    assert config.host == "127.0.0.1"
    assert config.maximum_observation_delay_seconds == 10.0
    with pytest.raises(ValueError):
        IBKRObserverConfig(host="broker.example.com")


def test_observer_is_not_broker_and_public_protocol_has_no_order_or_account_surface() -> None:
    assert not issubclass(IBKRObserver, Broker)
    allowed = {
        "connect",
        "disconnect",
        "request_server_time",
        "resolve_stock_contract",
        "capture_top_of_book_snapshot",
        "cancel_market_data",
        "api_tws_version",
    }
    public = {
        name for name, _ in inspect.getmembers(IBKRObservabilityClient) if not name.startswith("_")
    }
    assert public == allowed


def test_observability_package_does_not_import_order_models_or_call_forbidden_methods() -> None:
    package = Path("packages/stocker_execution/src/stocker_execution/ibkr_observability")
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(package.glob("*.py")))

    assert "stocker_execution.orders" not in source
    for forbidden_call in (
        ".placeOrder(",
        ".cancelOrder(",
        ".reqIds(",
        ".reqOpenOrders(",
        ".reqExecutions(",
        ".reqPositions(",
        ".reqAccountSummary(",
        ".reqAccountUpdates(",
        ".reqGlobalCancel(",
    ):
        assert forbidden_call not in source


def test_observer_sources_contain_no_credentials_network_side_effects_or_universe_dependency() -> (
    None
):
    package = Path("packages/stocker_execution/src/stocker_execution/ibkr_observability")
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(package.glob("*.py")))
    universe_source = Path(
        "packages/stocker_research/src/stocker_research/observable_event_ranking_v1/universe.py"
    ).read_text(encoding="utf-8")

    assert not re.search(r"(?:DU|U)\d{7,}", source)
    assert "password" not in source.lower()
    assert "socket.connect" not in source
    assert "stocker_execution.ibkr_observability" not in universe_source
    assert "ibkr" not in universe_source.lower()


def test_quote_classifications_never_treat_partial_frozen_delayed_or_stale_as_live_complete() -> (
    None
):
    assert classify_snapshot(_snapshot()) is ObservationClassification.LIVE_TOP_OF_BOOK_OBSERVED
    assert classify_snapshot(_snapshot(ask=None)) is ObservationClassification.LIVE_PARTIAL_QUOTE
    assert (
        classify_snapshot(_snapshot(snapshot_complete=False))
        is ObservationClassification.LIVE_PARTIAL_QUOTE
    )
    assert (
        classify_snapshot(_snapshot(market_data_type=MarketDataType.FROZEN))
        is ObservationClassification.FROZEN_NON_CURRENT
    )
    assert (
        classify_snapshot(_snapshot(market_data_type=MarketDataType.DELAYED))
        is ObservationClassification.DELAYED_NON_EXECUTABLE
    )
    assert classify_snapshot(_snapshot(response_delay=10.01)) is ObservationClassification.STALE
    assert classify_snapshot(_snapshot(completion_delay=10.01)) is ObservationClassification.STALE


def test_fake_client_capture_is_bounded_cancelled_and_requests_no_account_data(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot()
    fake = FakeIBKRObservabilityClient(contracts={"AAPL": _contract()}, snapshots={7: snapshot})
    observer = IBKRObserver(
        client=fake,
        config=IBKRObserverConfig(enabled=True, client_id=719),
        clock=lambda: snapshot.requested_timestamp,
    )

    async def run() -> QuoteObservationRecord:
        await observer.connect()
        record = await observer.capture_quote(
            plan=ObservationPlanItem(
                observation_id="obs-1",
                event_id="event-1",
                decision_id="slate-1",
                decision_timestamp=snapshot.requested_timestamp - timedelta(minutes=5),
                planned_entry_reference_timestamp=snapshot.requested_timestamp,
                planned_exit_reference_timestamp=snapshot.requested_timestamp
                + timedelta(minutes=60),
                planned_observation_timestamp=snapshot.requested_timestamp,
                symbol="AAPL",
                con_id=265598,
                required_observation_type="live_top_of_book",
                maximum_collection_delay_seconds=10.0,
                completion_status="planned",
            ),
            contract=_contract(),
            request_id=7,
        )
        await observer.disconnect()
        return record

    record = asyncio.run(run())

    assert record.classification is ObservationClassification.LIVE_TOP_OF_BOOK_OBSERVED
    assert fake.cancelled_request_ids == [7]
    assert fake.account_requests == 0
    assert fake.order_requests == 0
    ledger_path = append_quote_observation(tmp_path / "quote-ledger", record)
    assert '"fill_claim":false' in ledger_path.read_text(encoding="utf-8")
    with pytest.raises(QuoteLedgerError):
        append_quote_observation(tmp_path / "quote-ledger", record)


def test_capture_retries_are_bounded_and_every_attempt_is_cancelled() -> None:
    fake = FakeIBKRObservabilityClient()
    observer = IBKRObserver(
        client=fake,
        config=IBKRObserverConfig(
            enabled=True,
            client_id=720,
            bounded_retries=2,
            maximum_requests_per_second=1_000,
        ),
        clock=lambda: planned,
    )
    planned = datetime(2025, 7, 7, 14, 5, tzinfo=UTC)

    async def run() -> object:
        await observer.connect()
        record = await observer.capture_quote(
            plan=ObservationPlanItem(
                observation_id="obs-timeout",
                event_id="event-timeout",
                decision_id="slate-timeout",
                decision_timestamp=planned - timedelta(minutes=5),
                planned_entry_reference_timestamp=planned,
                planned_exit_reference_timestamp=planned + timedelta(minutes=60),
                planned_observation_timestamp=planned,
                symbol="AAPL",
                con_id=265598,
                required_observation_type="live_top_of_book",
                maximum_collection_delay_seconds=10.0,
                completion_status="planned",
            ),
            contract=_contract(),
            request_id=9,
        )
        await observer.disconnect()
        return record

    record = asyncio.run(run())

    assert record.classification is ObservationClassification.ERROR
    assert fake.capture_attempts == [9, 9, 9]
    assert fake.cancelled_request_ids == [9, 9, 9]


def test_late_invocation_cannot_backfill_a_missed_observation_window() -> None:
    snapshot = _snapshot()
    fake = FakeIBKRObservabilityClient(snapshots={7: snapshot})
    planned = snapshot.requested_timestamp
    observer = IBKRObserver(
        client=fake,
        config=IBKRObserverConfig(enabled=True, client_id=721),
        clock=lambda: planned + timedelta(seconds=20),
    )

    async def run() -> object:
        await observer.connect()
        record = await observer.capture_quote(
            plan=ObservationPlanItem(
                observation_id="obs-late",
                event_id="event-late",
                decision_id="slate-late",
                decision_timestamp=planned - timedelta(minutes=5),
                planned_entry_reference_timestamp=planned,
                planned_exit_reference_timestamp=planned + timedelta(minutes=60),
                planned_observation_timestamp=planned,
                symbol="AAPL",
                con_id=265598,
                required_observation_type="live_top_of_book",
                maximum_collection_delay_seconds=10.0,
                completion_status="planned",
            ),
            contract=_contract(),
            request_id=7,
        )
        await observer.disconnect()
        return record

    record = asyncio.run(run())

    assert record.classification is ObservationClassification.UNAVAILABLE
    assert fake.capture_attempts == []


def test_observation_plan_contains_entry_and_exit_without_a_fill_claim() -> None:
    event = {
        "event_id": "event-1",
        "slate_id": "slate-1",
        "symbol": "AAPL",
        "con_id": 265598,
        "assigned_decision_time": datetime(2025, 7, 7, 14, 0, tzinfo=UTC),
        "planned_entry_reference_time": datetime(2025, 7, 7, 14, 5, tzinfo=UTC),
        "planned_exit_reference_time": datetime(2025, 7, 7, 15, 5, tzinfo=UTC),
    }

    plan = build_observation_plan([event])

    assert [item.required_observation_type for item in plan] == [
        "live_top_of_book_entry_reference",
        "live_top_of_book_exit_reference",
    ]
    assert all(item.maximum_collection_delay_seconds == 10.0 for item in plan)
    assert all("fill" not in item.required_observation_type for item in plan)
    assert all(item.decision_timestamp == event["assigned_decision_time"] for item in plan)
    assert all(
        item.planned_entry_reference_timestamp == event["planned_entry_reference_time"]
        for item in plan
    )
    assert all(
        item.planned_exit_reference_timestamp == event["planned_exit_reference_time"]
        for item in plan
    )


def test_official_api_status_records_supported_zip_install_blocker_not_pypi_substitution() -> None:
    status = official_api_status()

    assert status.distribution_source == "official_ibkr_tws_api_zip_or_msi_only"
    assert "PyPI" in status.installation_policy
    assert not status.provenance_verified
    assert not status.installed
    assert status.blocker is not None
