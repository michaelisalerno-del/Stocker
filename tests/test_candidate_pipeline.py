from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from stocker_core.candidate_selection import CandidateIdentity
from stocker_core.config import RunsConfig
from stocker_core.markets import CapBucket, MarketId, MarketUniverseSpec, get_market
from stocker_core.methods import LEGACY_SESSION_HARD, SESSION_HARD, content_hash
from stocker_core.runs import Environment, RunInstance, RunState
from stocker_core.universes import InstrumentReference, UniverseDefinition
from stocker_dashboard.universe_runs import UniverseRunBuilder
from stocker_execution.candidate_pipeline import CandidatePipeline, CandidateStore
from stocker_execution.ibkr import HistoricalBar
from stocker_execution.runtime import ExchangeSessionResolver, MarketSession, MarketSessionState


def setup_run(market_id=MarketId.US_ALL, count=533):
    market = get_market(market_id)
    universe = UniverseDefinition(
        universe_id=market.listing_membership or "CACHED",
        name="Saved eligible population",
        market_spec=MarketUniverseSpec(market_id=market_id, cap_bucket=CapBucket.ALL),
        members=tuple(
            InstrumentReference(symbol=f"S{i}", exchange="SMART", currency=market.currency)
            for i in range(1, count + 1)
        ),
    )
    config, run = UniverseRunBuilder().add(
        RunsConfig(universes=(universe,)),
        market_id=market_id,
        strategy_id=SESSION_HARD.method_id,
        strategy_version=SESSION_HARD.version,
        environment=Environment.PAPER,
    )
    instance = RunInstance(run, run.universe_snapshot, RunState.ACTIVE)
    session = ExchangeSessionResolver().resolve(run, datetime(2026, 9, 2, 12, tzinfo=UTC))
    return config, instance, session


class Provider:
    def __init__(self):
        self.calls = 0

    async def acquire(self, instance):
        self.calls += 1
        return tuple(
            CandidateIdentity(
                i,
                r.symbol,
                r.primary_exchange,
                r.exchange,
                r.currency,
                instance.config.market_id,
                r.security_type,
            )
            for i, r in enumerate(instance.universe.members, 1)
        ), []


class Source:
    def __init__(self):
        self.calls = []
        self.failed = False

    async def prefix(self, identity, expected, due):
        self.calls.append((identity.con_id, len(expected)))
        if self.failed:
            raise RuntimeError("no opening data entitlement")
        return tuple(
            HistoricalBar(t, 1, 1 + identity.con_id / 1000, 0.9, 1 + identity.con_id / 10000, 1)
            for t in expected
        )


async def drain(pipeline, instance, session, clock):
    await pipeline.advance(instance, session, clock[0])
    if pipeline.tasks:
        await asyncio.gather(*pipeline.tasks.values())
    return await pipeline.advance(instance, session, clock[0])


def test_persisted_stages_restart_and_no_later_refill(tmp_path):
    async def scenario():
        _, instance, session = setup_run()
        clock = [session.opens_at - timedelta(minutes=1)]
        store, provider, source = CandidateStore(tmp_path / "state.sqlite"), Provider(), Source()
        pipeline = CandidatePipeline(store, provider, source, lambda: clock[0])
        await drain(pipeline, instance, session, clock)
        assert not pipeline.result(instance, session.session).requests
        assert store.summary(instance.config.run_id, session.session)["broad_eligible"] == 533
        survivors = set(range(1, 534))
        for index, (minute, capacity) in enumerate([(5, 250), (10, 50), (15, 30)]):
            clock[0] = session.opens_at + timedelta(minutes=minute)
            await drain(pipeline, instance, session, clock)
            current = {
                r.con_id for r in store.population(instance.config.run_id, session.session, index)
            }
            assert len(current) == capacity and current <= survivors
            assert {c for c, n in source.calls if n == minute} == survivors
            survivors = current
            clock[0] += timedelta(seconds=1)
            pipeline = CandidatePipeline(
                CandidateStore(store.path), provider, source, lambda: clock[0]
            )
            await pipeline.advance(instance, session, clock[0])
        assert provider.calls == 1
        requests = pipeline.result(instance, session.session).requests
        assert {r.instrument.con_id for r in requests} == survivors
        before = store.details(instance.config.run_id, session.session, 2000, 0)
        clock[0] += timedelta(hours=1)
        await pipeline.advance(instance, session, clock[0])
        assert store.details(instance.config.run_id, session.session, 2000, 0) == before
        assert len(source.calls) == 533 + 250 + 50

    asyncio.run(scenario())


@pytest.mark.parametrize("minute", [5, 7, 10, 15, 60])
def test_late_start_never_reconstructs_selection(tmp_path, minute):
    async def scenario():
        _, instance, session = setup_run()
        clock = [session.opens_at + timedelta(minutes=minute)]
        provider, source = Provider(), Source()
        pipeline = CandidatePipeline(
            CandidateStore(tmp_path / "late.sqlite"), provider, source, lambda: clock[0]
        )
        result = await pipeline.advance(instance, session, clock[0])
        assert result.requests == ()
        assert result.ineligible[0].reason == "CANDIDATE_SELECTION_WINDOW_MISSED"
        assert provider.calls == 0 and not source.calls

    asyncio.run(scenario())


def test_restart_with_unfrozen_missed_stage_fails_closed(tmp_path):
    async def scenario():
        _, instance, session = setup_run()
        clock = [session.opens_at]
        store, provider, source = CandidateStore(tmp_path / "restart.sqlite"), Provider(), Source()
        original = CandidatePipeline(store, provider, source, lambda: clock[0])
        await drain(original, instance, session, clock)
        clock[0] += timedelta(minutes=6)
        resumed = CandidatePipeline(store, provider, source, lambda: clock[0])
        result = await resumed.advance(instance, session, clock[0])
        assert result.ineligible[0].reason == "CANDIDATE_SELECTION_WINDOW_MISSED"
        assert not source.calls

    asyncio.run(scenario())


def test_acquisition_failure_and_market_isolation(tmp_path):
    async def scenario():
        _, us, us_session = setup_run()
        _, uk, uk_session = setup_run(MarketId.UK_LSE)
        clock = [uk_session.opens_at]
        store, source = CandidateStore(tmp_path / "markets.sqlite"), Source()
        pipeline = CandidatePipeline(store, Provider(), source, lambda: clock[0])
        await drain(pipeline, uk, uk_session, clock)
        source.failed = True
        clock[0] += timedelta(minutes=5)
        await drain(pipeline, uk, uk_session, clock)
        assert store.summary(uk.config.run_id, uk_session.session)["reason"].startswith(
            "BROAD_OPENING_DATA_CAPACITY_UNRESOLVED"
        )
        assert not pipeline.result(uk, uk_session.session).requests
        source.failed = False
        clock[0] = us_session.opens_at
        await drain(pipeline, us, us_session, clock)
        assert store.summary(us.config.run_id, us_session.session)["state"] == "BROAD_ELIGIBLE"
        assert store.summary(uk.config.run_id, uk_session.session)["state"] == "DEGRADED"

    asyncio.run(scenario())


def test_active_prefix_spans_calendar_break_without_counting_closed_minutes():
    start = datetime(2026, 9, 2, 3, 55, tzinfo=UTC)
    reopening = start + timedelta(minutes=65)
    session = MarketSession(
        start.date(),
        MarketSessionState.ACTIVE_SESSION,
        start,
        reopening + timedelta(hours=2),
        (start, reopening, reopening + timedelta(minutes=5)),
    )
    prefix = session.minute_prefix(15)
    assert prefix[4] == start + timedelta(minutes=4)
    assert prefix[5] == reopening and len(prefix) == 15
    from stocker_core.candidate_selection import SESSION_HARD_CANDIDATE_RECIPE, candidate_value

    rows = tuple(HistoricalBar(t, 1, 1, 1, 1, 1) for t in prefix)
    assert (
        candidate_value(
            SESSION_HARD_CANDIDATE_RECIPE.stages[-1],
            rows,
            expected_prefix=prefix,
            as_of=prefix[-1] + timedelta(minutes=1),
        ).value
        == 0
    )


def test_new_spec_changes_only_candidate_and_operational_fields():
    for market in MarketId:
        old, new = LEGACY_SESSION_HARD.specification(market), SESSION_HARD.specification(market)
        changed = {k for k in old if old[k] != new[k]}
        assert changed == {"method_version", "universe_search", "ranking_capacity"}
        assert content_hash(old) != content_hash(new)
        assert (
            new["candidate_selection"]["recipe_id"] == "SESSION_HARD_RANGE5_250_RV10_50_RV15_30_V1"
        )
    config, instance, _ = setup_run()
    assert RunsConfig.model_validate(config.model_dump(mode="json")).runs[0] == instance.config


def test_real_runtime_gates_history_and_builds_state_only_from_rv30(tmp_path):
    from stocker_execution.strategy_factory import MethodServices
    from test_stage8_runtime import (
        FakeBroker,
        FakeFeatureService,
        MutableClock,
        TriggerContextProvider,
        _hv_run,
        _runtime,
    )

    async def scenario():
        _, instance, session = setup_run(MarketId.US_NASDAQ)
        clock = MutableClock()
        clock.now = session.opens_at - timedelta(minutes=1)
        run = _hv_run().model_copy(
            update={
                "universe_snapshot": instance.universe.model_copy(update={"universe_id": "NASDAQ"})
            }
        )
        instance = replace(instance, config=run, universe=run.universe_snapshot)
        provider, source = Provider(), Source()
        pipeline = CandidatePipeline(
            CandidateStore(tmp_path / "candidates.sqlite"), provider, source, lambda: clock.now
        )
        features = FakeFeatureService()

        class Entries:
            trade_errors = {}

            def __init__(self):
                self.prepared = set()

            def prepare_trades(self, stock):
                self.prepared.add(stock.con_id)

            def release_unused_trades(self, stocks):
                self.prepared.intersection_update(stocks)

            async def trades_for(self, instruments, signals):
                return {}

        entries = Entries()
        broker = FakeBroker()
        runtime = _runtime(
            tmp_path,
            broker,
            run,
            clock=clock,
            feature_service=features,
            context_provider=TriggerContextProvider(),
            entry_source=entries,
        )
        default = runtime._default_method_services
        services = replace(
            default,
            qualify=pipeline.qualify,
            universe_lifecycle=pipeline.advance,
            universe_ready=pipeline.ready,
            universe_status=pipeline.store.summary,
            stop_universe=pipeline.stop,
            prepare_history_on_ready=True,
        )
        assert isinstance(services, MethodServices)
        runtime._method_services = {run.strategy_version: services}
        runtime._qualify = pipeline.qualify
        await runtime.start()
        await asyncio.gather(*pipeline.tasks.values())
        await runtime.poll_once()
        assert not features.prepared and not entries.prepared and features.calls == 0
        for minute in [5, 10, 15]:
            clock.now = session.opens_at + timedelta(minutes=minute)
            await runtime.poll_once()
            await asyncio.gather(*pipeline.tasks.values())
            await runtime.poll_once()
            if minute < 15:
                assert not features.prepared and features.calls == 0
        selected = {
            r.instrument.con_id for r in pipeline.result(instance, session.session).requests
        }
        assert len(selected) == 30
        await asyncio.gather(*(task for task, _ in runtime._expected_move_tasks.values()))
        assert all(set(row[2]) == selected for row in features.prepared)
        clock.now = session.opens_at + timedelta(minutes=25)
        await runtime.poll_once()
        assert entries.prepared == selected
        clock.now = session.opens_at + timedelta(minutes=30)
        await runtime.poll_once()
        await asyncio.gather(*runtime._checkpoint_tasks.values())
        method = runtime._strategies[run.run_id]
        assert len(method.signals) == 30
        assert {s.underlying_con_id for s in method.signals} == selected
        assert {s.underlying_con_id for s in method.cohort_labels.values()} == selected
        assert broker.submitted == []
        await runtime.stop()

    asyncio.run(scenario())


def test_empty_history_response_is_missing_and_shared_while_transport_failure_propagates(tmp_path):
    from stocker_execution.candidate_pipeline import OpeningBarSource
    from stocker_execution.history import IbkrHistoryCache
    from stocker_execution.ibkr import IbkrConnection, IbkrError, IbkrHistoricalDataUnavailable

    class Broker(IbkrConnection):
        def __init__(self):
            self.calls = []

        async def historical_bars(self, instrument, **kwargs):
            self.calls.append(kwargs)
            if instrument.con_id == 1:
                raise IbkrHistoricalDataUnavailable("no bars")
            raise IbkrError("connection unavailable")

    async def scenario():
        _, instance, session = setup_run(count=2)
        identities, _ = await Provider().acquire(instance)
        broker = Broker()
        source = OpeningBarSource(broker, IbkrHistoryCache(tmp_path / "history.sqlite"))
        prefix = session.minute_prefix(5)
        due = prefix[-1] + timedelta(minutes=1)
        assert await asyncio.gather(
            *(source.prefix(identities[0], prefix, due) for _ in range(2))
        ) == [(), ()]
        assert len(broker.calls) == 1
        assert broker.calls[0]["bar_size"] == "1 min"
        assert broker.calls[0]["regular_trading_hours"]
        assert broker.calls[0]["end_time"] == due
        with pytest.raises(IbkrError, match="connection unavailable"):
            await source.prefix(identities[1], prefix, due)

    asyncio.run(scenario())


def test_partial_broker_qualification_failure_never_becomes_a_smaller_broad_universe(tmp_path):
    from stocker_execution.candidate_pipeline import ConfiguredUniverseProvider
    from stocker_execution.ibkr import IbkrError, QualifiedInstrument

    class Broker:
        async def resolve_stock(self, symbol, **kwargs):
            if symbol == "S3":
                raise IbkrError("Gateway disconnected")
            return QualifiedInstrument(symbol, int(symbol[1:]), "SMART", "NASDAQ", "USD", "STK")

        async def qualify_discovery_candidate(self, row):
            stock = QualifiedInstrument(
                row.symbol,
                row.con_id,
                row.exchange,
                row.primary_exchange,
                row.currency,
                row.security_type,
            )
            return stock, "ETF" if row.symbol == "S2" else "COMMON"

    async def scenario():
        _, instance, session = setup_run(count=3)
        clock = [session.opens_at]
        pipeline = CandidatePipeline(
            CandidateStore(tmp_path / "state.sqlite"),
            ConfiguredUniverseProvider(Broker()),
            Source(),
            lambda: clock[0],
        )
        # A failed broad population must never need a second transaction to become degraded.
        separate_failures = []
        pipeline.store.fail = lambda *args: separate_failures.append(args)
        await drain(pipeline, instance, session, clock)
        assert not separate_failures
        summary = pipeline.store.summary(instance.config.run_id, session.session)
        assert summary["broad_eligible"] == 1
        assert summary["state"] == "DEGRADED"
        assert "BROAD_OPENING_DATA_CAPACITY_UNRESOLVED" in summary["reason"]
        assert not pipeline.result(instance, session.session).requests
        identity = pipeline.store.population(instance.config.run_id, session.session)[0]
        assert identity.listing_exchange == "NASDAQ" and identity.routing_exchange == "SMART"
        with pipeline.store.connect() as db:
            reasons = [
                json.loads(r[0])
                for r in db.execute("SELECT payload FROM opening_candidate_rejections")
            ]
        assert sum(r["acquisition_failure"] for r in reasons) == 1
        resumed = CandidatePipeline(CandidateStore(pipeline.store.path), Provider(),
                                    Source(), lambda: clock[0])
        assert not (await resumed.advance(instance, session, clock[0])).requests

    asyncio.run(scenario())


def test_failed_final_stage_and_rows_are_atomic_and_cannot_activate_on_restart(tmp_path):
    async def scenario():
        _, instance, session = setup_run()
        clock = [session.opens_at]
        source = Source()
        store = CandidateStore(tmp_path / "atomic.sqlite")
        pipeline = CandidatePipeline(store, Provider(), source, lambda: clock[0])
        await drain(pipeline, instance, session, clock)
        for minute in (5, 10):
            clock[0] = session.opens_at + timedelta(minutes=minute)
            await drain(pipeline, instance, session, clock)
        source.failed = True
        clock[0] = session.opens_at + timedelta(minutes=15)
        await drain(pipeline, instance, session, clock)
        row = store.summary(instance.config.run_id, session.session)
        assert row["completed_stages"] == 3 and row["state"] == "DEGRADED"
        resumed = CandidatePipeline(
            CandidateStore(store.path), Provider(), Source(), lambda: clock[0]
        )
        assert not (await resumed.advance(instance, session, clock[0])).requests
        assert not resumed.ready(instance.config.run_id, session.session)

    asyncio.run(scenario())
