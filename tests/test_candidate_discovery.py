"""Discovery fixtures contain no historical trading outcomes and need no Gateway."""

import asyncio
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from legacy_discovery_support import UniverseRunBuilder
from stocker_core.config import IbkrConfig, RunsConfig
from stocker_core.discovery import SESSION_HARD_DISCOVERY, DiscoveryProfile, UniverseSource
from stocker_core.markets import CAP_BUCKETS_V1, CapBucket, MarketId, get_market
from stocker_core.methods import LEGACY_SESSION_HARD as SESSION_HARD
from stocker_core.methods import content_hash
from stocker_core.runs import Environment, RunConfig, RunInstance, RunState
from stocker_core.universes import InstrumentReference
from stocker_execution.activity_shortlist import ScannerCapabilities
from stocker_execution.discovery import (
    CandidateDiscovery,
    DiscoveryRow,
    DiscoveryStore,
    discovery_summary,
    scanner_requests,
    watch_identities,
)
from stocker_execution.ibkr import IbkrConnection, IbkrError, QualifiedInstrument
from stocker_execution.session_hard_universe import SessionHardUniverseSearch
from stocker_execution.stage5 import Stage5Analyzer, Stage5FeatureResult, Stage5Status
from test_stage10_extension_builder import empty_config

NOW = datetime(2026, 9, 8, 14, 0, tzinfo=UTC)
MARKET = get_market(MarketId.US_ALL)
FILTERS = frozenset(
    {
        "marketCapAbove1e6",
        "marketCapBelow1e6",
        "priceAbove",
        "usdPriceAbove",
        "volumeAbove",
        "avgVolumeAbove",
    }
)
CAPABILITIES = ScannerCapabilities(
    frozenset({MARKET.scanner_location}),
    frozenset({"TOP_TRADE_RATE"}),
    FILTERS,
)


def make_run(**profile_overrides):
    config, run = UniverseRunBuilder().add(
        empty_config(),
        market_id="US_ALL",
        strategy_id=SESSION_HARD.method_id,
        strategy_version=SESSION_HARD.version,
        environment=Environment.PAPER,
    )
    if profile_overrides:
        profile = DiscoveryProfile.model_validate(
            SESSION_HARD_DISCOVERY.model_dump() | profile_overrides,
        )
        run = RunConfig.model_validate(run.model_dump() | {"discovery_profile": profile})
    return config, run


def row(con_id=1, **changes):
    return replace(
        DiscoveryRow(
            con_id,
            f"NEW{con_id}",
            "SMART",
            "NASDAQ",
            "USD",
            "STK",
            0,
            {"distance": "raw"},
        ),
        **changes,
    )


class Broker(IbkrConnection):
    def __init__(self, rows=None):
        self.rows = rows or {}
        self.scans = []
        self.resolved = []
        self.capabilities = CAPABILITIES
        self.inflight = 0
        self.peak = 0
        self.bad_types = {}
        self.bad_contracts = set()

    async def scanner_capabilities(self):
        return self.capabilities

    async def discovery_scan(self, request):
        self.scans.append(request)
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        await asyncio.sleep(0)
        self.inflight -= 1
        result = self.rows.get(request.cap_band, ())
        if isinstance(result, Exception):
            raise result
        return result

    async def qualify_discovery_candidate(self, candidate):
        self.resolved.append(candidate.con_id)
        if candidate.con_id in self.bad_contracts:
            raise IbkrError("INVALID_CONTRACT: fixture")
        return QualifiedInstrument(
            **{key: asdict(candidate)[key] for key in QualifiedInstrument.__dataclass_fields__}
        ), self.bad_types.get(candidate.con_id, "COMMON")

    async def resolve_stock(self, *args, **kwargs):
        raise AssertionError("Dynamic identities must not be resolved again by symbol")

    async def historical_bars(self, *args, **kwargs):
        raise AssertionError("Discovery must not fetch historical data")

    async def submit_protected_order(self, *args, **kwargs):
        raise AssertionError("Discovery must not submit orders")


def discover(tmp_path, broker, run=None):
    run = run or make_run()[1]
    store = DiscoveryStore(tmp_path / "audit.sqlite")
    result = asyncio.run(
        CandidateDiscovery(broker, store).discover(
            run,
            MARKET,
            NOW.date(),
            NOW,
            NOW,
        )
    )
    return result, store


def test_five_canonical_bands_are_independent_and_bounded(tmp_path):
    broker = Broker(
        {
            band: (row(index + 1),)
            for index, band in enumerate(
                SESSION_HARD_DISCOVERY.cap_bands,
            )
        }
    )
    result, _ = discover(tmp_path, broker)
    assert result["status"] == "READY"
    assert [r.cap_band for r in broker.scans] == list(SESSION_HARD_DISCOVERY.cap_bands)
    assert broker.peak == 2
    assert len(watch_identities(result)) == 5
    for request in broker.scans:
        definition = CAP_BUCKETS_V1.definition(request.cap_band)
        assert request.minimum_cap_millions == definition.scanner_minimum_millions
        assert request.maximum_cap_millions == definition.scanner_maximum_millions
        assert request.scanner == "TOP_TRADE_RATE"
    assert [r.minimum_cap_millions for r in broker.scans] == [50, 300, 2000, 10000, 200000]
    assert broker.scans[-1].maximum_cap_millions is None


def test_merge_deduplicates_conid_even_when_symbols_differ_and_preserves_provenance(tmp_path):
    broker = Broker(
        {
            CapBucket.MICRO: (row(123), row(124, symbol="SAME", raw_rank=1)),
            CapBucket.SMALL: (row(123, symbol="RENAMED", raw_rank=3), row(125, symbol="SAME")),
        }
    )
    document, store = discover(tmp_path, broker)
    assert len(document["observations"]) == 4
    assert len(document["candidates"]) == 3
    assert sorted(broker.resolved) == [123, 124, 125]
    candidate = next(c for c in document["candidates"] if c["identity"]["con_id"] == 123)
    origins = [document["observations"][i] for i in candidate["observation_indices"]]
    assert {r["cap_band"] for r in origins} == {CapBucket.MICRO, CapBucket.SMALL}
    assert {r["raw_rank"] for r in origins} == {0, 3}
    assert all(r["metadata"]["distance"] == "raw" for r in origins)
    assert origins[-1]["rejection_reason"] == "DUPLICATE_CONID"
    assert DiscoveryStore(store.path).history(make_run()[1].run_id)[0] == document


def test_rejections_are_explicit_and_invalid_rows_never_reach_contract_work(tmp_path):
    broker = Broker(
        {
            CapBucket.MICRO: (
                row(0),
                row(2, security_type="OPT"),
                row(3, currency="EUR"),
                row(4),
                row(5),
                row(6),
            )
        }
    )
    broker.bad_types[4] = "ETF"
    broker.bad_contracts.add(5)
    document, _ = discover(tmp_path, broker)
    assert broker.resolved == [4, 5, 6]
    assert len(watch_identities(document)) == 1
    reasons = {c["identity"]["con_id"]: c["rejection_reason"] for c in document["candidates"]}
    assert reasons == {
        0: "INVALID_CONTRACT",
        2: "WRONG_SECURITY_TYPE",
        3: "INVALID_CONTRACT",
        4: "WRONG_SECURITY_TYPE",
        5: "INVALID_CONTRACT",
        6: "",
    }
    assert discovery_summary(document)["rejected_candidates"] == 5


def test_limits_interleave_cap_bands_and_limit_expensive_work(tmp_path):
    broker = Broker(
        {
            band: tuple(row(index * 100 + rank + 1, raw_rank=rank) for rank in range(50))
            for index, band in enumerate(SESSION_HARD_DISCOVERY.cap_bands)
        }
    )
    _, run = make_run(merged_candidate_limit=10, monitoring_limit=5)
    document, _ = discover(tmp_path, broker, run)
    assert len(document["observations"]) == 250
    assert len(broker.resolved) == 10
    assert {i.con_id for i in watch_identities(document)} == {1, 101, 201, 301, 401}
    assert discovery_summary(document)["rejection_counts"] == {"RESOURCE_LIMIT": 245}


def test_default_pool_can_exceed_fifty(tmp_path):
    broker = Broker(
        {
            band: tuple(row(index * 100 + rank + 1, raw_rank=rank) for rank in range(50))
            for index, band in enumerate(SESSION_HARD_DISCOVERY.cap_bands)
        }
    )
    document, _ = discover(tmp_path, broker)
    assert len(watch_identities(document)) == 150


def test_surplus_broker_rows_are_audited_without_exceeding_requested_band_limit(tmp_path):
    broker = Broker({CapBucket.MICRO: tuple(row(i + 1, raw_rank=i) for i in range(5))})
    _, run = make_run(results_per_band=2)
    document, _ = discover(tmp_path, broker, run)
    assert len(document["observations"]) == 5
    assert len(watch_identities(document)) == 2
    assert broker.resolved == [1, 2]
    assert discovery_summary(document)["rejection_counts"] == {"RESOURCE_LIMIT": 3}


def test_failed_attempt_reuse_requires_explicit_refresh(tmp_path):
    async def scenario():
        broker = Broker({CapBucket.MICRO: IbkrError("Scanner unavailable")})
        _, run = make_run()
        store = DiscoveryStore(tmp_path / "failure.sqlite")
        service = CandidateDiscovery(broker, store)
        failed = await service.discover(run, MARKET, NOW.date(), NOW, NOW)
        broker.rows = {CapBucket.MICRO: (row(),)}
        assert await service.discover(run, MARKET, NOW.date(), NOW, NOW) == failed
        assert len(broker.scans) == 5
        store.request_refresh(run.run_id)
        ready = await service.discover(run, MARKET, NOW.date(), NOW, NOW)
        assert ready["status"] == "READY"
        assert len(store.history(run.run_id)) == 2

    asyncio.run(scenario())


def test_interrupted_attempt_does_not_promote_a_partial_watchlist(tmp_path):
    document, store = discover(tmp_path, Broker({CapBucket.MICRO: (row(),)}))
    document["status"] = "RUNNING"
    store.save(document)
    broker = Broker()
    recovered = asyncio.run(
        CandidateDiscovery(broker, store).discover(
            make_run()[1],
            MARKET,
            NOW.date(),
            NOW,
            NOW,
        )
    )
    assert recovered["status"] == "FAILED"
    assert recovered["reason"] == "DISCOVERY_INTERRUPTED"
    assert not watch_identities(recovered)
    assert broker.scans == []


@pytest.mark.parametrize("unsupported", ["scanner", "location", "filters", "instrument"])
def test_invalid_broker_configuration_is_persisted_before_any_scan(tmp_path, unsupported):
    broker = Broker()
    broker.capabilities = {
        "scanner": replace(CAPABILITIES, scan_codes=frozenset({"HOT_BY_VOLUME"})),
        "location": replace(CAPABILITIES, locations=frozenset()),
        "filters": replace(CAPABILITIES, filters=FILTERS - {"avgVolumeAbove"}),
        "instrument": replace(
            CAPABILITIES,
            location_instruments={
                MARKET.scanner_location: frozenset({"FUT"}),
            },
        ),
    }[unsupported]
    document, store = discover(tmp_path, broker)
    assert document["status"] == "FAILED"
    assert "SCANNER_NOT_SUPPORTED" in document["reason"]
    assert broker.scans == []
    assert store.history(make_run()[1].run_id)[0]["status"] == "FAILED"
    assert not watch_identities(document)


def test_empty_results_and_partial_failure_never_fall_back_to_fixed_members(tmp_path):
    empty, _ = discover(tmp_path, Broker())
    assert empty["status"] == "EMPTY"
    assert not watch_identities(empty)
    _, run = make_run()
    run = run.model_copy(update={"run_id": "partial"})
    broker = Broker({CapBucket.MICRO: (row(),), CapBucket.SMALL: IbkrError("Unavailable")})
    partial, store = discover(tmp_path, broker, run)
    assert partial["status"] == "FAILED"
    assert len(partial["observations"]) == 1
    assert not broker.resolved
    assert not watch_identities(partial)
    assert store.history("partial")[0]["segments"][1]["reason"] == "Unavailable"


def test_refresh_new_session_and_new_run_preserve_attempts_without_constant_rescans(tmp_path):
    async def scenario():
        _, run = make_run()
        broker = Broker({CapBucket.MICRO: (row(),)})
        store = DiscoveryStore(tmp_path / "audit.sqlite")
        service = CandidateDiscovery(broker, store)
        first = await service.discover(run, MARKET, NOW.date(), NOW, NOW)
        repeated = await asyncio.gather(
            *(service.discover(run, MARKET, NOW.date(), NOW, NOW) for _ in range(3))
        )
        assert all(r == first for r in repeated)
        assert len(broker.scans) == 5
        store.request_refresh(run.run_id)
        rebuilt = await service.discover(run, MARKET, NOW.date(), NOW, NOW)
        assert rebuilt["discovery_id"] != first["discovery_id"]
        next_day = NOW + timedelta(days=1)
        tomorrow = await service.discover(run, MARKET, next_day.date(), next_day, next_day)
        assert tomorrow["session"] != first["session"]
        other = await service.discover(
            run.model_copy(update={"run_id": "independent"}),
            MARKET,
            NOW.date(),
            NOW,
            NOW,
        )
        assert other["discovery_id"] != first["discovery_id"]
        assert len(broker.scans) == 20
        assert len(store.history(run.run_id)) == 3

    asyncio.run(scenario())


def test_dynamic_identities_reach_existing_pre_interface_without_scanner_score(tmp_path):
    async def scenario():
        config, run = make_run()
        assert not run.universe_snapshot.members  # No saved-listing dependency.
        instance = RunInstance(run, config.universes[-1], RunState.ACTIVE)
        broker = Broker({CapBucket.MICRO: (row(456, raw_rank=49), row(123, raw_rank=0))})
        search = SessionHardUniverseSearch(broker, tmp_path / "audit.sqlite", lambda: NOW)
        qualified = await search.qualify((instance,))
        assert {r.instrument.con_id for r in qualified.requests} == {123, 456}
        assert not qualified.ineligible
        seen = []

        class ExistingFeatureBoundary:
            async def get_feature(self, instrument, *, session, t0):
                seen.append(instrument)
                return Stage5FeatureResult(
                    instrument.con_id,
                    instrument.symbol,
                    session,
                    t0,
                    Stage5Status.PRE_CONTEXT_NOT_READY,
                    "required history missing",
                )

        analyzer = Stage5Analyzer(ExistingFeatureBoundary())
        snapshots = await analyzer.analyze(qualified.requests, session=NOW.date(), t0=NOW)
        assert {i.con_id for i in seen} == {123, 456}
        assert all(s.status is Stage5Status.PRE_CONTEXT_NOT_READY for s in snapshots)
        assert all(
            set(asdict(i))
            == {
                "con_id",
                "symbol",
                "exchange",
                "primary_exchange",
                "currency",
                "security_type",
            }
            for i in seen
        )
        assert all(r.memberships[0].run_id == run.run_id for r in qualified.requests)

    asyncio.run(scenario())


@pytest.mark.parametrize("source", [UniverseSource.FIXED, UniverseSource.RESEARCH])
def test_explicit_fixed_sources_converge_on_the_same_stage5_interface(tmp_path, source):
    async def scenario():
        config, run = make_run()
        universe = run.universe_snapshot.model_copy(
            update={
                "members": (InstrumentReference(symbol="FIXED", exchange="SMART", currency="USD"),)
            }
        )
        run = RunConfig.model_validate(
            run.model_dump()
            | {
                "universe_source": source,
                "discovery_profile": None,
                "universe_snapshot": universe,
            }
        )

        class FixedBroker(Broker):
            async def resolve_stock(self, *args, **kwargs):
                return QualifiedInstrument("FIXED", 777, "SMART", "NASDAQ", "USD", "STK")

        broker = FixedBroker()
        search = SessionHardUniverseSearch(broker, tmp_path / "fixed.sqlite", lambda: NOW)
        result = await search.qualify((RunInstance(run, universe, RunState.ACTIVE),))
        assert result.requests[0].instrument.con_id == 777
        assert broker.scans == []

    asyncio.run(scenario())


def test_scheduled_discovery_validates_early_and_runs_only_when_due(tmp_path):
    async def scenario():
        config, run = make_run()
        broker = Broker({CapBucket.MICRO: (row(),)})
        clock = [NOW.replace(hour=13, minute=30)]
        search = SessionHardUniverseSearch(broker, tmp_path / "audit.sqlite", lambda: clock[0])
        instance = RunInstance(run, config.universes[-1], RunState.ACTIVE)
        pending = await search.qualify((instance,))
        assert pending.ineligible[0].reason == "SCHEDULED"
        assert not broker.scans
        before = search.discovery.store.history(run.run_id)[0]
        clock[0] = NOW
        result = await search.qualify((instance,))
        assert len(result.requests) == 1
        after = search.discovery.store.history(run.run_id)[0]
        assert before["discovery_id"] == after["discovery_id"]
        assert after["captured_at"] == NOW.isoformat()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"results_per_band": 51},
        {"monitoring_limit": 251},
        {"cap_bands": ["ALL"]},
        {"minimum_price": float("nan")},
        {"cap_bands": ["SMALL", "SMALL"]},
        {"unexpected_filter": True},
    ],
)
def test_profile_validation(changes):
    with pytest.raises(ValidationError):
        DiscoveryProfile.model_validate(SESSION_HARD_DISCOVERY.model_dump() | changes)


def test_method_owns_policy_but_resource_settings_are_configurable():
    _, run = make_run(monitoring_limit=200, minimum_price=0.5)
    assert run.discovery_profile.monitoring_limit == 200
    assert run.method_spec_hash == content_hash(run.method_spec)
    with pytest.raises(ValidationError, match="belong"):
        RunConfig.model_validate(
            run.model_dump()
            | {
                "discovery_profile": run.discovery_profile.model_dump()
                | {"scanner": "HOT_BY_VOLUME"},
            }
        )


def test_us_builder_needs_no_seeded_universe():
    config, run = UniverseRunBuilder().add(
        RunsConfig(),
        market_id="US_ALL",
        strategy_id=SESSION_HARD.method_id,
        strategy_version=SESSION_HARD.version,
        environment=Environment.PAPER,
    )
    assert run.uses_dynamic_discovery
    assert config.universes == (run.universe_snapshot,)
    assert not config.universes[0].members


def test_frozen_trading_specification_is_unchanged():
    # Captured from pre-discovery commit 4074729, excluding discovery/version/capacity wording.
    spec = SESSION_HARD.specification(MarketId.US_ALL)
    frozen = {
        key: value
        for key, value in spec.items()
        if key not in {"universe_search", "method_version", "ranking_capacity"}
    }
    assert (
        content_hash(frozen) == "4dd7c400cd28225eb736717bd85fcad74f8488328652131f7510335144b2d2bf"
    )


def test_broker_adapter_preserves_raw_rows_uses_native_cap_units_and_caches_conid():
    class Client:
        def __init__(self):
            self.calls = 0
            self.requests = []
            self.details = SimpleNamespace(
                contract=SimpleNamespace(
                    conId=123,
                    symbol="NEW",
                    secType="STK",
                    exchange="SMART",
                    primaryExchange="NASDAQ",
                    currency="USD",
                ),
                stockType="COMMON",
            )

        def isConnected(self):
            return True

        async def reqScannerDataAsync(self, subscription, options, filters):
            self.requests.append((subscription, filters))
            return [
                SimpleNamespace(rank=0, contractDetails=self.details),
                SimpleNamespace(rank=1, contractDetails=self.details),
            ]

        async def reqContractDetailsAsync(self, contract):
            assert contract.conId == 123
            self.calls += 1
            return [self.details]

    async def scenario():
        client = Client()
        connection = IbkrConnection(
            IbkrConfig(
                environment=Environment.PAPER,
                host="127.0.0.1",
                port=4002,
                client_id=19,
            ),
            client=client,
        )
        connection._account_id = "DU_FIXTURE"
        request = scanner_requests(SESSION_HARD_DISCOVERY, MARKET, CAPABILITIES)[0]
        rows = await connection.discovery_scan(request)
        assert len(rows) == 2  # Never discard duplicate observations at the broker boundary.
        subscription, filters = client.requests[0]
        assert subscription.marketCapAbove == 50
        assert subscription.marketCapBelow == 300
        assert subscription.abovePrice > 1e100  # Local-price filter remains unset.
        assert subscription.aboveVolume == 1000
        assert [(f.tag, f.value) for f in filters] == [
            ("avgVolumeAbove", "100000"), ("usdPriceAbove", "1"),
        ]
        assert await connection.qualify_discovery_candidate(rows[0]) == (
            await connection.qualify_discovery_candidate(rows[1])
        )
        assert client.calls == 1

    asyncio.run(scenario())
