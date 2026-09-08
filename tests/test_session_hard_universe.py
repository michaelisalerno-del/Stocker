import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from stocker_core.markets import ActivityScanner, CapBucket, MarketId, get_market
from stocker_core.methods import SESSION_HARD
from stocker_core.runs import Environment, RunInstance, RunState
from stocker_dashboard.universe_runs import UniverseRunBuilder
from stocker_execution.activity_shortlist import ScannerCandidate, ScannerCapabilities
from stocker_execution.ibkr import IbkrConnection, QualifiedInstrument
from stocker_execution.session_hard_universe import SessionHardUniverseSearch
from test_stage10_extension_builder import empty_config


class MarketBroker(IbkrConnection):
    def __init__(self, market):
        self.market = market
        self.config = SimpleNamespace(environment=Environment.PAPER)
        self.scans = []

    async def scanner_capabilities(self):
        return ScannerCapabilities(
            frozenset({self.market.scanner_location}),
            frozenset(s.value for s in ActivityScanner),
            frozenset(),
        )

    async def activity_scan(self, *, market, cap_bucket, component, max_results):
        self.scans.append((market.market_id, cap_bucket, component, max_results))
        exchange = "LSE" if market.market_id is MarketId.UK_LSE else "ASX"
        return (ScannerCandidate(component, 1, "TEST", 123, "SMART", exchange, market.currency),)

    async def resolve_stock(self, symbol, *, exchange, currency, primary_exchange=None):
        return QualifiedInstrument(symbol, 123, exchange, primary_exchange, currency, "STK")

    async def submit_protected_order(self, *_args):
        raise AssertionError("Universe testing must never submit orders")


@pytest.mark.parametrize("market_id", [MarketId.UK_LSE, MarketId.AUSTRALIA_ASX])
def test_method_uses_existing_local_session_scan_and_reloads_snapshot(tmp_path, market_id):
    async def scenario():
        config, run = UniverseRunBuilder().add(
            empty_config(),
            market_id=market_id,
            strategy_id=SESSION_HARD.method_id,
            strategy_version=SESSION_HARD.version,
            environment=Environment.PAPER,
        )
        instance = RunInstance(run, config.universes[-1], RunState.ACTIVE)
        market = get_market(market_id)
        opening = datetime.combine(
            datetime(2026, 9, 8).date(),
            market.regular_sessions[0].opens_at,
            ZoneInfo(market.timezone),
        ).astimezone(UTC)
        now = [opening]
        broker = MarketBroker(market)
        database = tmp_path / "universe.sqlite3"
        search = SessionHardUniverseSearch(broker, database, lambda: now[0])
        scheduled = await search.qualify((instance,))
        assert scheduled.ineligible[0].reason == "SCHEDULED"
        assert broker.scans == []
        now[0] = opening + timedelta(minutes=15)
        result = await search.qualify((instance,))
        assert (
            len(result.requests) == 1 and result.requests[0].instrument.currency == market.currency
        )
        assert len(broker.scans) == 3
        assert all(cap is CapBucket.ALL and limit == 50 for _, cap, _, limit in broker.scans)
        replay = SessionHardUniverseSearch(broker, database, lambda: now[0])
        assert await replay.qualify((instance,)) == result
        assert len(broker.scans) == 3
        now[0] = opening + timedelta(days=1)
        assert (await replay.qualify((instance,))).ineligible[0].reason == "SCHEDULED"

    asyncio.run(scenario())
