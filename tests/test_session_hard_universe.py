import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from stocker_core.markets import ActivityScanner, CapBucket, MarketId, get_market
from stocker_core.methods import SESSION_HARD
from stocker_core.runs import ACTIVITY_CAPACITY_V2_VERSION, Environment, RunInstance, RunState
from stocker_dashboard.universe_runs import UniverseRunBuilder
from stocker_execution.activity_shortlist import ScannerCandidate, ScannerCapabilities
from stocker_execution.ibkr import IbkrConnection, QualifiedInstrument
from stocker_execution.session_hard_universe import SessionHardUniverseSearch
from test_stage10_extension_builder import empty_config


class MarketBroker(IbkrConnection):
    def __init__(self, market):
        self.market = market
        self.config = SimpleNamespace(environment=Environment.PAPER, market_data_line_budget=100)
        self.scans = []
        self.resolved = []

    async def scanner_capabilities(self):
        return ScannerCapabilities(
            frozenset({self.market.scanner_location}),
            frozenset(s.value for s in ActivityScanner),
            frozenset(),
        )

    async def activity_scan(self, *, market, cap_bucket, component, max_results):
        self.scans.append((market.market_id, cap_bucket, component, max_results))
        return tuple(
            ScannerCandidate(component, i + 1, f"TEST{i}", 123 + i, "SMART", None, market.currency)
            for i in range(50)
        )

    async def resolve_stock(self, symbol, *, exchange, currency, primary_exchange=None):
        self.resolved.append(symbol)
        return QualifiedInstrument(
            symbol, 123 + int(symbol[4:]), exchange, primary_exchange, currency, "STK"
        )

    async def submit_protected_order(self, *_args):
        raise AssertionError("Universe testing must never submit orders")


@pytest.mark.parametrize("market_id", list(MarketId))
def test_method_uses_existing_local_session_scan_and_reloads_snapshot(tmp_path, market_id):
    async def scenario():
        from stocker_core.universes import InstrumentReference

        initial = empty_config()
        initial = initial.model_copy(
            update={
                "universes": tuple(
                    u.model_copy(
                        update={
                            "members": tuple(
                                InstrumentReference(
                                    symbol=f"TEST{i}", exchange="SMART", currency="USD"
                                )
                                for i in range(50)
                            )
                        }
                    )
                    for u in initial.universes
                )
            }
        )
        config, run = UniverseRunBuilder().add(
            initial,
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
        # A restart after the old one-minute capture window must screen now,
        # never backdate the observation or process the entire listing universe.
        now[0] = opening + timedelta(minutes=17)
        result = await search.qualify((instance,))
        assert (
            len(result.requests) == 50 and result.requests[0].instrument.currency == market.currency
        )
        assert set(broker.resolved) == {f"TEST{i}" for i in range(50)}
        assert len(broker.scans) == 3
        assert all(cap is CapBucket.ALL and limit == 50 for _, cap, _, limit in broker.scans)
        snapshot = search.activity.store.get(
            market_id.value,
            CapBucket.ALL,
            now[0].astimezone(ZoneInfo(market.timezone)).date(),
            profile_id="ACTIVITY_CAPACITY_V2",
            profile_version=ACTIVITY_CAPACITY_V2_VERSION,
        )
        assert snapshot.screen_timestamp == now[0]
        assert len(snapshot.candidates) == 50
        assert sum(c.selected for c in snapshot.candidates) == 50
        now[0] += timedelta(minutes=5)
        replay = SessionHardUniverseSearch(broker, database, lambda: now[0])
        assert await replay.qualify((instance,)) == result
        assert len(broker.scans) == 3
        now[0] = opening + timedelta(days=1)
        assert (await replay.qualify((instance,))).ineligible[0].reason == "SCHEDULED"

    asyncio.run(scenario())


def test_us_listing_membership_is_applied_before_ranking_and_qualification(tmp_path):
    from stocker_core.universes import InstrumentReference
    from test_stage10_extension_builder import add

    config, run = add(market=MarketId.US_NASDAQ)
    universe = config.universes[-1].model_copy(
        update={"members": (InstrumentReference(symbol="TEST8", exchange="SMART", currency="USD"),)}
    )
    broker = MarketBroker(get_market(MarketId.US_NASDAQ))
    search = SessionHardUniverseSearch(
        broker, tmp_path / "screen.sqlite", lambda: datetime(2026, 9, 8, 14, tzinfo=UTC)
    )
    result = asyncio.run(search.qualify((RunInstance(run, universe, RunState.ACTIVE),)))
    assert [r.instrument.symbol for r in result.requests] == ["TEST8"]
    assert broker.resolved == ["TEST8"]


def test_150_scanner_hits_are_bounded_to_50_before_contract_work(tmp_path):
    from test_stage10_extension_builder import add

    class DisjointScans(MarketBroker):
        async def activity_scan(self, *, market, cap_bucket, component, max_results):
            offset = list(ActivityScanner).index(component) * 50
            return tuple(
                ScannerCandidate(
                    component, i + 1, f"TEST{offset + i}", 123 + offset + i,
                    "SMART", None, market.currency,
                )
                for i in range(max_results)
            )

        def prepare_trade_events(self, instrument):
            raise AssertionError("Screening must not allocate live trade streams")

    config, run = add(market=MarketId.UK_LSE)
    broker = DisjointScans(get_market(MarketId.UK_LSE))
    now = datetime(2026, 9, 8, 8, tzinfo=UTC)
    search = SessionHardUniverseSearch(broker, tmp_path / "screen.sqlite", lambda: now)
    result = asyncio.run(
        search.qualify((RunInstance(run, config.universes[-1], RunState.ACTIVE),))
    )
    snapshot = search.activity.store.get(
        run.market_id.value, CapBucket.ALL, now.date(),
        profile_id="ACTIVITY_CAPACITY_V2", profile_version=ACTIVITY_CAPACITY_V2_VERSION,
    )
    assert len(snapshot.candidates) == 150
    selected = {c.symbol for c in snapshot.candidates if c.selected}
    assert len(selected) == len(result.requests) == len(broker.resolved) == 50
    assert set(broker.resolved) == selected
    assert run.method_spec["universe_search"]["watch_limit"] == 50


def test_screen_size_is_independent_of_feed_budget_and_scanner_failures_stay_isolated(tmp_path):
    from stocker_execution.ibkr import IbkrError
    from test_stage10_extension_builder import add

    config, run = add(market=MarketId.UK_LSE)
    instance = RunInstance(run, config.universes[-1], RunState.ACTIVE)
    broker = MarketBroker(get_market(MarketId.UK_LSE))
    broker.config.market_data_line_budget = 40

    def now():
        return datetime(2026, 9, 8, 8, tzinfo=UTC)

    search = SessionHardUniverseSearch(broker, tmp_path / "small.sqlite", now)
    result = asyncio.run(search.qualify((instance,)))
    assert len(result.requests) == 50 and len(broker.resolved) == 50

    class Unentitled(MarketBroker):
        async def activity_scan(self, **kwargs):
            raise IbkrError("Market data permission unavailable")

    denied = Unentitled(broker.market)
    search = SessionHardUniverseSearch(denied, tmp_path / "denied.sqlite", now)
    result = asyncio.run(search.qualify((instance,)))
    assert not result.requests and not denied.resolved
    assert result.ineligible[0].reason == "DATA_NOT_ENTITLED"


def test_dashboard_reads_current_profile_not_legacy_shortlist(tmp_path):
    from stocker_execution.activity_shortlist import ActivityShortlistStore
    from test_stage10_dashboard import _seed_authoritative_state
    from test_stage10_extension_builder import add

    reads = _seed_authoritative_state(tmp_path)
    config, run = add(market=MarketId.UK_LSE)
    now = datetime(2026, 9, 8, 8, tzinfo=UTC)
    reads.config = config
    reads.clock = lambda: now
    search = SessionHardUniverseSearch(
        MarketBroker(get_market(MarketId.UK_LSE)), tmp_path / "activity.sqlite", lambda: now
    )
    asyncio.run(search.qualify((RunInstance(run, config.universes[-1], RunState.ACTIVE),)))
    reads.activity_store = ActivityShortlistStore(tmp_path / "activity.sqlite")
    assert reads.runs()[0]["candidate_count"] == 50
    detail = reads.run_detail(run.run_id)
    assert detail["watchlist_size"] == 50
    assert detail["activity_screen"]["profile_id"] == "ACTIVITY_CAPACITY_V2"
    assert len(detail["activity_screen"]["candidates"]) == 50


def test_five_stock_snapshot_is_preserved_but_not_reused(tmp_path):
    from stocker_execution.activity_shortlist import (
        ActivityShortlistService,
        ActivityShortlistStore,
    )
    from test_stage10_extension_builder import add

    config, run = add(market=MarketId.UK_LSE)
    now = datetime(2026, 9, 8, 8, tzinfo=UTC)
    broker = MarketBroker(get_market(MarketId.UK_LSE))
    path = tmp_path / "activity.sqlite"
    store = ActivityShortlistStore(path)
    legacy = ActivityShortlistService(
        store, profile_id="ACTIVITY_CAPACITY_V2",
        profile_version="ACTIVITY_CAPACITY_V2_WARNINGS",
        watch_limit=5, allow_late_capture=True
    )
    original = asyncio.run(
        legacy.get_or_create(
            broker,
            market=broker.market,
            cap_bucket=CapBucket.ALL,
            session=now.date(),
            screen_at=now,
            now=now,
        )
    )
    assert sum(c.selected for c in original.candidates) == 5
    current = SessionHardUniverseSearch(broker, path, lambda: now)
    result = asyncio.run(
        current.qualify((RunInstance(run, config.universes[-1], RunState.ACTIVE),))
    )
    assert len(result.requests) == 50 and len(broker.scans) == 6
    assert (
        store.get(
            run.market_id.value,
            CapBucket.ALL,
            now.date(),
            profile_id="ACTIVITY_CAPACITY_V2",
            profile_version="ACTIVITY_CAPACITY_V2_WARNINGS",
        )
        == original
    )


@pytest.mark.parametrize("market_id", [MarketId.HONG_KONG_HKEX, MarketId.JAPAN_TSE])
def test_method_uses_exact_active_prefix_across_lunch_break(tmp_path, market_id):
    from dataclasses import replace

    from stocker_execution.history import IbkrHistoryCache
    from stocker_execution.ibkr import HistoricalBar
    from stocker_execution.runtime import ExchangeSessionResolver
    from stocker_execution.session_hard_data import SOURCE, IbkrSessionDataSource
    from test_stage6_session_hard_strategy import ready_snapshot
    from test_stage10_extension_builder import add

    _, run = add(market=market_id)
    session = ExchangeSessionResolver().resolve(run, datetime(2026, 9, 8, 4, tzinfo=UTC))
    checkpoint = 32
    t0 = session.active_bar_starts[checkpoint]
    minutes = tuple(
        start + timedelta(minutes=i)
        for start in session.active_bar_starts[:checkpoint]
        for i in range(5)
    )
    assert t0 - minutes[0] > timedelta(minutes=checkpoint * 5)
    bars = tuple(
        HistoricalBar(t, 100 + i / 100, 101 + i / 100, 99 + i / 100, 100.5 + i / 100, 100)
        for i, t in enumerate(minutes)
    )

    class HistoryBroker(MarketBroker):
        async def historical_bars(self, _instrument, **kwargs):
            self.request_duration = int(kwargs["duration"].split()[0])
            return bars

    broker = HistoryBroker(get_market(market_id))
    source = IbkrSessionDataSource(broker, IbkrHistoryCache(tmp_path / "history.sqlite"))
    row = replace(
        ready_snapshot(pre_move_m=1),
        session=session.session,
        t0=t0,
        expected_move_source=SOURCE,
        historical_volatility=0.2,
    )
    instrument = QualifiedInstrument(
        row.symbol, row.con_id, "SMART", None, get_market(market_id).currency, "STK"
    )
    result = asyncio.run(source.context_for(run, (row,), checkpoint, {row.con_id: instrument}, ()))
    assert len(result.whipsaw_features) == 1
    assert broker.request_duration >= (t0 - minutes[0]).total_seconds()
