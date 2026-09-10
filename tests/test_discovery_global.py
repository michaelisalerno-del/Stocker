"""Global discovery currency, capability and migration regressions. No broker required."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from stocker_core.markets import MarketId, get_market
from stocker_core.methods import SESSION_HARD, content_hash
from stocker_execution.discovery import CandidateDiscovery, DiscoveryStore, scanner_requests
from stocker_execution.ibkr import CurrentQuote, IbkrError
from test_candidate_discovery import CAPABILITIES, FILTERS
from test_ibkr_resources import (
    CallbackEvent,
    LowLevelScannerClient,
    connected_stream_boundary,
    wait_until,
)
from test_session_hard_universe import MarketBroker
from test_stage10_extension_builder import add


@pytest.mark.parametrize("currency,rate", [("GBP", 0.8), ("JPY", 150), ("HKD", 8)])
def test_canonical_usd_caps_translate_to_native_currency(currency, rate):
    from dataclasses import replace

    market = next(get_market(m) for m in MarketId if get_market(m).currency == currency)
    profile = SESSION_HARD.discovery_profile(market.market_id)
    capabilities = replace(CAPABILITIES, locations=frozenset({profile.scanner_location}))
    requests = scanner_requests(profile, market, capabilities, local_per_usd=rate)
    assert [r.minimum_cap_millions for r in requests] == [
        bound * rate for bound in (50, 300, 2000, 10000, 200000)
    ]
    assert [r.maximum_cap_millions for r in requests] == [
        300 * rate, 2000 * rate, 10000 * rate, 200000 * rate, None,
    ]
    assert all(r.price_currency == "USD" and r.cap_currency == currency for r in requests)


def test_unsupported_market_fails_before_fx_or_scans_and_preserves_attempt(tmp_path):
    from dataclasses import replace

    _, run = add(market=MarketId.SOUTH_AFRICA_JSE)
    broker = MarketBroker(get_market(run.market_id))
    async def capabilities():
        return replace(CAPABILITIES, locations=frozenset())
    async def no_fx(currency):
        raise AssertionError("Unsupported location must fail before FX work")
    broker.scanner_capabilities = capabilities
    broker.discovery_fx = no_fx
    now = datetime.now(UTC)
    store = DiscoveryStore(tmp_path / "audit.sqlite")
    result = asyncio.run(CandidateDiscovery(broker, store).discover(
        run, broker.market, now.date(), now, now,
    ))
    assert result["status"] == "FAILED"
    assert result["reason"].startswith("SCANNER_NOT_SUPPORTED")
    assert not broker.scans
    assert store.history(run.run_id) == [result]


@pytest.mark.parametrize("code,message,empty", [
    (165, "Historical Market Data Service query message:no items retrieved", True),
    (165, "Another historical query problem", False),
    (162, "Market Scanner is not configured for one of the chosen locations.", False),
])
def test_scanner_distinguishes_empty_bands_from_broker_failure(code, message, empty):
    async def scenario():
        client = LowLevelScannerClient()
        client.errorEvent = CallbackEvent()
        broker = connected_stream_boundary(client)
        request = scanner_requests(
            SESSION_HARD.discovery_profile(MarketId.US_ALL),
            get_market(MarketId.US_ALL), CAPABILITIES,
        )[0]
        task = asyncio.create_task(broker.discovery_scan(request))
        await wait_until(lambda: len(client.wrapper.futures), 1)
        client.errorEvent.emit(71, code, message, None)
        client.wrapper.futures[0].set_result([])
        if empty:
            assert await task == ()
        else:
            with pytest.raises(IbkrError):
                await task
        assert client.cancelled_scanners == [71]
    asyncio.run(scenario())


@pytest.mark.parametrize("currency,bid,ask,expected", [
    ("JPY", 149, 151, 150), ("GBP", 1.2, 1.3, 0.8),
])
def test_broker_fx_uses_auditable_two_sided_snapshot(currency, bid, ask, expected):
    async def scenario():
        client = LowLevelScannerClient()
        async def details(contract):
            contract.conId = 456
            return [SimpleNamespace(contract=contract)]
        client.reqContractDetailsAsync = details
        broker = connected_stream_boundary(client)
        now = datetime.now(UTC)
        async def quote(instrument):
            assert instrument.security_type == "CASH" and instrument.con_id == 456
            return CurrentQuote(instrument.symbol, 456, now, bid, ask, None, None, 1)
        broker.current_quote = quote
        fx = await broker.discovery_fx(currency)
        assert fx.local_per_usd == expected
        assert fx.bid == bid and fx.ask == ask and fx.observed_at == now.isoformat()
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["missing", "stale", "delayed", "crossed"])
def test_fx_failure_cannot_create_a_watch_pool(tmp_path, fault):
    async def scenario():
        client = LowLevelScannerClient()
        async def details(contract):
            contract.conId = 456
            return [SimpleNamespace(contract=contract)]
        client.reqContractDetailsAsync = details
        broker = connected_stream_boundary(client)
        now = datetime.now(UTC)
        async def quote(instrument):
            return CurrentQuote(
                "GBP", 456, now - timedelta(minutes=5) if fault == "stale" else now,
                None if fault == "missing" else 1.3, 1.2 if fault == "crossed" else 1.4,
                None, None, 3 if fault == "delayed" else 1,
            )
        broker.current_quote = quote
        _, run = add(market=MarketId.UK_LSE)
        from stocker_execution.activity_shortlist import ScannerCapabilities
        broker._scanner_capabilities = ScannerCapabilities(
            frozenset({run.discovery_profile.scanner_location}),
            frozenset({"TOP_TRADE_RATE"}), FILTERS,
        )
        store = DiscoveryStore(tmp_path / "audit.sqlite")
        result = await CandidateDiscovery(broker, store).discover(
            run, get_market(run.market_id), now.date(), now, now,
        )
        assert result["status"] == "FAILED"
        assert "MARKET_DATA_UNAVAILABLE" in result["reason"]
        assert not result["observations"] and not result["candidates"]
        assert store.history(run.run_id) == [result]
    asyncio.run(scenario())


@pytest.mark.parametrize("market", [MarketId.US_ALL, MarketId.UK_LSE, MarketId.AUSTRALIA_ASX])
def test_v6_migration_archives_original_and_preserves_operational_limits(market):
    from test_activity_filter_migration import migration
    config, run = add(market=market)
    payload = config.model_dump(mode="json")
    previous = payload["runs"][0]
    previous.update(run_id="old", strategy_version=migration().PREVIOUS_VERSION)
    previous["method_spec"]["method_version"] = migration().PREVIOUS_VERSION
    previous["method_spec_hash"] = content_hash(previous["method_spec"])
    previous["discovery_profile"]["monitoring_limit"] = 200
    upgraded, mapping = migration().migrate(payload)
    old, new = upgraded.runs
    assert old.archived and not old.enabled and new.enabled
    assert new.uses_dynamic_discovery and not new.universe_snapshot.members
    assert new.discovery_profile.monitoring_limit == 200
    assert mapping == {"old": new.run_id}
    for field in ("data_requirements", "qualification", "vetoes", "direction", "entry",
                  "exits", "economics", "execution", "session", "artifact_hashes"):
        assert new.method_spec[field] == old.method_spec[field]
