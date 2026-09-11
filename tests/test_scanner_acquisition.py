"""Prospective acquisition boundaries; every broker is fake."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from stocker_core.acquisition import ACQUISITION_EXPERIMENT_V1, AcquisitionRecipe
from stocker_core.candidate_selection import CandidateIdentity
from stocker_core.markets import MarketId, get_market
from stocker_execution.acquisition_store import AcquisitionStore
from stocker_execution.activity_shortlist import ScannerCapabilities
from stocker_execution.candidate_oracle import (
    OracleAudit,
    recall_metrics,
    reconstruct,
    transport_parity,
)
from stocker_execution.candidate_pipeline import CandidatePipeline, CandidateStore
from stocker_execution.discovery import DiscoveryRow
from stocker_execution.ibkr import HistoricalBar, QualifiedInstrument
from stocker_execution.scanner_acquisition import ScannerAcquisition, acquisition_scans
from test_candidate_pipeline import Source, drain, setup_run

# Codes advertised by Gateway 178; individual test brokers remain entirely fake.
CODES = {
    "TOP_TRADE_RATE",
    "TOP_VOLUME_RATE",
    "HOT_BY_VOLUME",
    "TOP_OPEN_PERC_GAIN",
    "TOP_OPEN_PERC_LOSE",
}


def capabilities(market_id=MarketId.US_ALL):
    market = get_market(market_id)
    return ScannerCapabilities(
        frozenset({market.scanner_location}),
        frozenset(CODES),
        frozenset({"marketCapAbove", "marketCapBelow"}),
    )


class Broker:
    def __init__(self, market_id=MarketId.US_ALL):
        self.market_id = market_id
        self.requests = []
        self.active = self.maximum = self.qualifications = 0
        self.failure = None
        self.audit_history_lock = asyncio.Lock()

    def resource_status(self):
        return SimpleNamespace(pending_historical_work=0)

    async def scanner_capabilities(self):
        return capabilities(self.market_id)

    async def discovery_fx(self, currency):
        return SimpleNamespace(local_per_usd=0.75)

    async def acquisition_scan(self, request, audit):
        self.requests.append(request)
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(0)
            if self.failure == request.family:
                raise RuntimeError("component entitlement error")
            market = get_market(self.market_id)
            # Two duplicated observations and one component-specific observation.
            n = sorted(CODES).index(request.scan_code) + 2
            rows = tuple(
                DiscoveryRow(i, f"S{i}", "SMART", None, market.currency, "STK", rank, {})
                for rank, i in enumerate((1, n, 1))
            )
            audit.update(
                row_count=len(rows), latency_ms=2, warnings=["IBKR scanner precision warning (492)"]
            )
            return rows
        finally:
            self.active -= 1

    async def qualify_discovery_candidate(self, row):
        self.qualifications += 1
        await asyncio.sleep(0)
        return QualifiedInstrument(
            row.symbol, row.con_id, "SMART", None, row.currency, "STK"
        ), "COMMON"

    async def resolve_stock(self, symbol, **kwargs):
        return QualifiedInstrument(
            symbol, int(symbol[1:]), "SMART", None, kwargs["currency"], "STK"
        )


class WideBroker(Broker):
    async def acquisition_scan(self, request, audit):
        index = ACQUISITION_EXPERIMENT_V1.cap_slices.index(request.cap_slice)
        return tuple(
            DiscoveryRow(i, f"S{i}", "SMART", None, "USD", "STK", rank, {})
            for rank, i in enumerate(range(index * 50 + 1, index * 50 + 51))
        )


@pytest.mark.parametrize("market_id", [MarketId.US_ALL, MarketId.UK_LSE, MarketId.AUSTRALIA_ASX])
def test_observed_gateway_since_open_codes_exclude_overnight_gaps(market_id):
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/scanner_acquisition/gateway_178_opening_codes.json"
        ).read_text()
    )
    caps = ScannerCapabilities(
        locations=frozenset(fixture["locations"]),
        scan_codes=frozenset(fixture["scan_codes"]),
        filters=frozenset(fixture["filters"]),
        scan_descriptions=fixture["scan_descriptions"],
    )
    plans = acquisition_scans(ACQUISITION_EXPERIMENT_V1, get_market(market_id), caps, 1.0)
    assert len(plans) == 35 and all(not p.unsupported_reason for p in plans)
    assert {p.scan_code for p in plans if p.family == "OPENING_PERCENT_GAIN"} == {
        "TOP_OPEN_PERC_GAIN"
    }
    assert {p.scan_code for p in plans if p.family == "OPENING_PERCENT_LOSS"} == {
        "TOP_OPEN_PERC_LOSE"
    }
    assert not {"HIGH_OPEN_GAP", "LOW_OPEN_GAP"} & {p.scan_code for p in plans}
    missing = replace(caps, scan_codes=caps.scan_codes - {"TOP_OPEN_PERC_GAIN"})
    failed = acquisition_scans(ACQUISITION_EXPERIMENT_V1, get_market(market_id), missing, 1.0)
    assert all(
        p.unsupported_reason and not p.scan_code
        for p in failed
        if p.family == "OPENING_PERCENT_GAIN"
    )
    assert all(not p.unsupported_reason for p in failed if p.family != "OPENING_PERCENT_GAIN")
    scoped = replace(
        caps,
        location_scan_codes={
            get_market(market_id).scanner_location: caps.scan_codes - {"TOP_OPEN_PERC_LOSE"}
        },
    )
    restricted = acquisition_scans(ACQUISITION_EXPERIMENT_V1, get_market(market_id), scoped, 1.0)
    assert all(
        p.unsupported_reason and not p.scan_code
        for p in restricted
        if p.family == "OPENING_PERCENT_LOSS"
    )


def test_capabilities_define_exact_components_and_floorless_coverage():
    plans = acquisition_scans(
        ACQUISITION_EXPERIMENT_V1, get_market(MarketId.US_ALL), capabilities()
    )
    assert len(plans) == 35
    assert {p.scan_code for p in plans} == CODES
    assert not any(p.unsupported_reason for p in plans)
    assert all(p.filters == () for p in plans if p.cap_slice == "UNCAPPED")
    assert all(
        p.filters == (("marketCapBelow", "50.0"),) for p in plans if p.cap_slice == "BELOW_MICRO"
    )
    unavailable = replace(capabilities(), scan_codes=frozenset({"TOP_TRADE_RATE"}))
    failed = acquisition_scans(ACQUISITION_EXPERIMENT_V1, get_market(MarketId.US_ALL), unavailable)
    assert all(p.unsupported_reason for p in failed if p.family != "TOP_TRADE_RATE")
    assert not any(p.scan_code for p in failed if p.family.startswith("OPENING"))


@pytest.mark.parametrize("market_id", [MarketId.US_ALL, MarketId.UK_LSE, MarketId.AUSTRALIA_ASX])
def test_union_persistence_deadlines_and_market_isolation(tmp_path, market_id):
    async def scenario():
        _, instance, session = setup_run(market_id, 10)
        now = [session.opens_at]
        observed = []

        async def wait(due):
            observed.append(due)
            now[0] = max(now[0], due)

        store = AcquisitionStore(tmp_path / "acquisition.sqlite")
        broker = Broker(market_id)
        provider = ScannerAcquisition(broker, store, lambda: now[0], wait)
        pool, errors = await provider.acquire(instance)
        assert not errors
        assert len(pool) == 6
        assert {i.market for i in pool} == {market_id}
        assert broker.maximum == 2
        assert broker.qualifications == 6
        assert len(broker.requests) == 105
        assert observed == [session.minute_prefix(15)[m] for m in (1, 3, 4)]
        summary = store.summary(instance.config.run_id, session.session)
        assert summary["raw_hits"] == 315 and summary["duplicate_hits"] == 309
        assert summary["acquisition_count"] == 6
        assert summary["range5_broker_requests"] == 0
        assert summary["upstream_evidence"] == (
            "PROSPECTIVE_IBKR_TEST"
            if market_id is MarketId.US_ALL
            else "UNVALIDATED_CROSS_MARKET_ACQUISITION_TRANSFER"
        )
        assert "targets" not in summary["oracle_metrics"]
        now[0] = session.closes_at
        restored = ScannerAcquisition(broker, AcquisitionStore(store.path), lambda: now[0], wait)
        assert (await restored.acquire(instance))[0] == pool
        assert len(broker.requests) == 105
        assert store.summary("unrelated-run", session.session) is None
        with pytest.raises(ValueError, match="sealed"):
            store.add_pool(instance.config.run_id, session.session, pool[0], 0, 3, now[0], 0)
        hits = store.details(instance.config.run_id, session.session, "hits", 2, 1)
        assert len(hits) == 2 and hits[0]["scanner_rank"] == 1
        assert "range_5m_rank" not in hits[0]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["TOP_TRADE_RATE", "late"])
def test_failed_or_late_acquisition_never_falls_back(tmp_path, failure):
    async def scenario():
        _, instance, session = setup_run(count=10)
        now = [session.opens_at]

        async def wait(due):
            now[0] = due + (timedelta(minutes=5) if failure == "late" else timedelta())

        broker = Broker()
        broker.failure = failure
        store = AcquisitionStore(tmp_path / "failure.sqlite")
        provider = ScannerAcquisition(broker, store, lambda: now[0], wait)
        _, errors = await provider.acquire(instance)
        assert errors[0]["acquisition_failure"]
        assert store.summary(instance.config.run_id, session.session)["state"] in {
            "SCANNER_ACQUISITION_PARTIAL",
            "SCANNER_ACQUISITION_FAILED",
        }
        before = len(broker.requests)
        await provider.acquire(instance)
        assert len(broker.requests) == before

    asyncio.run(scenario())


def test_acquisition_receives_full_membership_but_range_only_receives_union(tmp_path):
    async def scenario():
        _, instance, session = setup_run(count=533)
        now = [session.opens_at]

        async def wait(due):
            now[0] = due

        acquisition = AcquisitionStore(tmp_path / "state.sqlite")
        provider = ScannerAcquisition(Broker(), acquisition, lambda: now[0], wait)
        source = Source()
        pipeline = CandidatePipeline(
            CandidateStore(acquisition.path), provider, source, lambda: now[0]
        )
        await drain(pipeline, instance, session, now)
        assert (
            acquisition.summary(instance.config.run_id, session.session)["broad_membership"] == 533
        )
        for minute in (5, 10, 15):
            now[0] = session.opens_at + timedelta(minutes=minute)
            await drain(pipeline, instance, session, now)
        assert {i for i, m in source.calls if m == 5} == {1, 2, 3, 4, 5, 6}
        assert len(pipeline.result(instance, session.session).requests) == 6
        before = pipeline.store.details(instance.config.run_id, session.session, 2000, 0)
        # Delayed full-population oracle contains names that were never acquired.
        for i, ref in enumerate(instance.universe.members, 1):
            identity = CandidateIdentity(
                i, ref.symbol, None, "SMART", "USD", MarketId.US_ALL, "STK"
            )
            bars = await source.prefix(identity, session.minute_prefix(15), now[0])
            acquisition.audit_progress(
                instance.config.run_id, session.session, i - 1, identity, "ELIGIBLE", bars
            )
        metadata = json.loads(
            acquisition.session(instance.config.run_id, session.session)["metadata"]
        )
        oracle = OracleAudit(Broker(), acquisition, source, lambda: now[0])
        oracle.finish(
            instance.config.run_id,
            session.session,
            metadata,
            acquisition.oracle_rows(instance.config.run_id, session.session),
        )
        assert pipeline.store.details(instance.config.run_id, session.session, 2000, 0) == before
        result = acquisition.summary(instance.config.run_id, session.session)
        assert result["oracle_state"] == "COMPLETE"
        assert result["oracle_metrics"]["recipes"]["ACTIVE"]["RANGE250"]["captured"] == 0
        misses = acquisition.details(instance.config.run_id, session.session, "misses", 20, 0)
        assert len(misses) == 20 and misses[0]["oracle_rank"] == 1
        assert len(acquisition.recall_history(instance.config.run_id)) > 0

    asyncio.run(scenario())


def test_oracle_recall_buckets_contributions_and_five_minute_transport():
    _, instance, session = setup_run(count=300)
    population = tuple(
        CandidateIdentity(i, f"S{i}", None, "SMART", "USD", MarketId.US_ALL, "STK")
        for i in range(1, 301)
    )
    prefix = session.minute_prefix(15)
    bars = {
        i.con_id: tuple(
            HistoricalBar(t, 1, 1 + i.con_id / 1000, 0.9, 1 + i.con_id / 10000, 1) for t in prefix
        )
        for i in population
    }
    ranked = reconstruct(population, bars, prefix, MarketId.US_ALL)
    assert [sum(r.selected for r in stage) for stage in ranked] == [250, 50, 30]
    acquired = {r.identity.con_id for r in ranked[0] if r.selected} - {300}
    hits = [{"con_id": i, "component": "a", "sweep": 0, "scanner_rank": 7} for i in acquired]
    metrics = recall_metrics(
        ranked,
        acquired,
        acquired,
        hits,
        {"a": {"family": "TOP_TRADE_RATE", "cap_slice": "UNCAPPED"}},
        {},
    )
    assert metrics["recipes"]["ACTIVE"]["RANGE250"]["recall"] == 249 / 250
    assert metrics["range_rank_buckets"]["1-25"] == {"available": 25, "captured": 24}
    assert metrics["component_contributions"]["RANGE250:a"]["unique"] == 249
    five = {i.con_id: (replace(bars[i.con_id][0]),) for i in population}
    parity = transport_parity(population, bars, five, prefix, MarketId.US_ALL)
    assert parity["exact_scores"] == 300 and parity["same_top250"]
    five[300] = (replace(five[300][0], high=1.01),)
    assert not transport_parity(population, bars, five, prefix, MarketId.US_ALL)["same_ordering"]


def test_recipe_changes_require_new_version_and_immutable_saved_sessions(tmp_path):
    _, instance, session = setup_run(count=1)
    recipe = ACQUISITION_EXPERIMENT_V1.model_dump(mode="json")
    store = AcquisitionStore(tmp_path / "versions.sqlite")
    store.begin("a", session.session, {"recipe": recipe}, instance.universe.members)
    with pytest.raises(ValueError, match="new recipe ID"):
        store.begin(
            "b",
            session.session,
            {"recipe": recipe | {"rows_per_component": 10}},
            instance.universe.members,
        )
    assert AcquisitionRecipe(sweep_active_seconds=(60,)).sweep_active_seconds == (60,)
    with pytest.raises(ValueError, match="uncapped"):
        AcquisitionRecipe(cap_slices=("MICRO",))


def test_oracle_yields_and_restarts_without_trading_writes(tmp_path):
    async def scenario():
        _, instance, session = setup_run(count=2)
        now = [session.opens_at]

        async def wait(due):
            now[0] = due

        store = AcquisitionStore(tmp_path / "oracle.sqlite")
        broker = Broker()
        await ScannerAcquisition(broker, store, lambda: now[0], wait).acquire(instance)
        oracle = OracleAudit(broker, store, Source(), lambda: now[0])
        await oracle.tick(instance, True)
        await oracle.task
        assert store.summary(instance.config.run_id, session.session)["oracle_completed"] == 0
        now[0] = session.closes_at
        await oracle.tick(instance, False)
        assert oracle.task.done()
        await oracle.tick(instance, True)
        await oracle.task
        assert store.summary(instance.config.run_id, session.session)["oracle_completed"] == 1
        restored = OracleAudit(broker, AcquisitionStore(store.path), Source(), lambda: now[0])
        for _ in range(3):
            await restored.tick(instance, True)
            if restored.task:
                await restored.task
        assert store.summary(instance.config.run_id, session.session)["oracle_state"] == "COMPLETE"
        assert store.next_oracle_row(instance.config.run_id, session.session) is None

    asyncio.run(scenario())


@pytest.mark.parametrize("warning_request", [71, 999])
def test_actual_boundary_capability_cache_data_end_and_cancellation(tmp_path, warning_request):
    from test_ibkr_resources import (
        CallbackEvent,
        LowLevelScannerClient,
        connected_stream_boundary,
        wait_until,
    )

    async def scenario():
        client = LowLevelScannerClient()
        client.errorEvent = CallbackEvent()
        client.client.serverVersion = lambda: 178
        client.wrapper.scannerData = lambda *args: None
        client.wrapper.scannerDataEnd = lambda req_id: client.wrapper.futures[0].set_result(
            [
                SimpleNamespace(
                    rank=0,
                    contractDetails=SimpleNamespace(
                        contract=SimpleNamespace(
                            conId=1,
                            symbol="S1",
                            exchange="SMART",
                            primaryExchange="NASDAQ",
                            currency="USD",
                            secType="STK",
                        )
                    ),
                )
            ]
        )
        broker = connected_stream_boundary(client)
        a, b = await asyncio.gather(broker.scanner_capabilities(), broker.scanner_capabilities())
        assert a is b and client.parameter_requests == 1
        assert a.raw_xml and a.retrieved_at and a.server_version == "178"
        store = AcquisitionStore(tmp_path / "caps.sqlite")
        assert store.save_capabilities(asdict(a)) == store.save_capabilities(asdict(b))
        plan = next(
            p
            for p in acquisition_scans(ACQUISITION_EXPERIMENT_V1, get_market(MarketId.US_ALL), a)
            if p.family == "TOP_TRADE_RATE" and p.cap_slice == "UNCAPPED"
        )
        audit = {}
        task = asyncio.create_task(broker.acquisition_scan(plan, audit))
        await wait_until(lambda: len(client.wrapper.futures), 1)
        client.errorEvent.emit(warning_request, 492, "precise results permission", None)
        client.wrapper.scannerData(71, 0)
        client.wrapper.scannerDataEnd(71)
        result = await task
        assert result[0].con_id == 1
        assert bool(audit["warnings"]) == (warning_request == 71)
        assert (
            audit["request_start"]
            <= audit["first_response"]
            <= audit["scanner_data_end"]
            <= audit["cancelled_at"]
        )
        assert client.cancelled_scanners == [71]
        assert not broker._scanner_timings

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["cancel", "pacing", "no_data"])
def test_history_transport_cancellation_errors_are_not_flat_stock(outcome):
    from stocker_execution.ibkr import IbkrError, IbkrHistoricalDataUnavailable
    from test_ibkr_resources import (
        CallbackEvent,
        LowLevelScannerClient,
        connected_stream_boundary,
        wait_until,
    )

    async def scenario():
        client = LowLevelScannerClient()
        client.errorEvent = CallbackEvent()
        cancellations = []
        client.client.cancelHistoricalData = cancellations.append

        async def history(contract, **kwargs):
            assert kwargs["timeout"] == 0
            return await client.wrapper.startReq(123, container=[])

        client.reqHistoricalDataAsync = history
        broker = connected_stream_boundary(client)
        task = asyncio.create_task(
            broker.historical_bars(
                QualifiedInstrument("S1", 1, "SMART", None, "USD", "STK"),
                bar_size="1 min",
                duration="300 S",
                what_to_show="TRADES",
                regular_trading_hours=True,
            )
        )
        await wait_until(lambda: len(client.wrapper.futures), 1)
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            client.errorEvent.emit(
                123,
                162,
                "pacing violation" if outcome == "pacing" else "query returned no data",
                None,
            )
            client.wrapper.futures[0].set_result([])
            with pytest.raises(IbkrError) as caught:
                await task
            assert isinstance(caught.value, IbkrHistoricalDataUnavailable) == (outcome == "no_data")
        assert cancellations == ([123] if outcome == "cancel" else [])
        assert broker.resource_status().pending_historical_work == 0

    asyncio.run(scenario())


def test_dashboard_acquisition_uses_compact_sql_projection_and_paginated_misses(tmp_path):
    from fastapi.testclient import TestClient

    from stocker_dashboard.app import create_dashboard_app
    from test_stage10_dashboard import _seed_authoritative_state

    reads = _seed_authoritative_state(tmp_path)
    config, instance, session = setup_run(count=533)
    reads.config = config
    reads.clock = lambda: session.opens_at
    status = reads.runtime_status()
    reads.runtime_status = lambda: replace(status, runs=())
    store = AcquisitionStore(reads.stage5_store.path)
    metadata = {
        "recipe": ACQUISITION_EXPERIMENT_V1.model_dump(mode="json"),
        "upstream_evidence": "PROSPECTIVE_IBKR_TEST",
    }
    store.begin(instance.config.run_id, session.session, metadata, instance.universe.members)
    with store.connect() as db:
        db.execute(
            "UPDATE acquisition_sessions SET oracle_metrics=?",
            (
                json.dumps(
                    {
                        "recipes": {"ACTIVE": {"RANGE250": {"recall": 0.988}}},
                        "targets": [{"captured": False, "con_id": i} for i in range(5000)],
                    }
                ),
            ),
        )
    client = TestClient(create_dashboard_app(reads, None))
    response = client.get("/api/runs")
    assert response.status_code == 200 and len(response.content) < 20000
    acquisition = response.json()[0]["acquisition"]
    assert acquisition["broad_membership"] == 533
    assert "targets" not in acquisition["oracle_metrics"]
    response = client.get(
        f"/api/runs/{instance.config.run_id}/acquisition",
        params={"session": str(session.session), "kind": "misses", "limit": 2, "offset": 3},
    )
    assert response.status_code == 200
    assert [r["con_id"] for r in response.json()["rows"]] == [3, 4]


def test_late_or_failed_component_hits_do_not_inflate_shadow_recall():
    from stocker_core.acquisition import SHADOW_RECIPES

    _, _, session = setup_run(count=1)
    identity = CandidateIdentity(1, "S1", None, "SMART", "USD", MarketId.US_ALL, "STK")
    prefix = session.minute_prefix(15)
    cutoff = prefix[4] + timedelta(minutes=1)
    bars = {1: tuple(HistoricalBar(t, 1, 2, 0.5, 1.5, 1) for t in prefix)}
    ranked = reconstruct((identity,), bars, prefix, MarketId.US_ALL)
    common = {"con_id": 1, "scanner_rank": 0, "sweep": 0, "component_status": "COMPLETE"}
    hits = [
        common | {"component": "activity", "observed_at": prefix[1].isoformat()},
        common | {"component": "opening", "observed_at": cutoff.isoformat()},
        common
        | {
            "component": "failed",
            "component_status": "FAILED",
            "observed_at": prefix[0].isoformat(),
        },
    ]
    components = {
        "activity": {"family": "TOP_TRADE_RATE", "cap_slice": "UNCAPPED"},
        "opening": {"family": "OPENING_PERCENT_GAIN", "cap_slice": "UNCAPPED"},
        "failed": {"family": "OPENING_PERCENT_GAIN", "cap_slice": "UNCAPPED"},
    }
    metrics = recall_metrics(ranked, {1}, {1}, hits, components, SHADOW_RECIPES, cutoff)
    assert metrics["recipes"]["OPENING_MOVEMENT_ONLY_V1"]["RANGE250"]["recall"] == 0
    assert metrics["targets"][0]["components"] == ["activity"]
    assert metrics["component_contributions"]["RANGE250:activity"]["unique"] == 1


def test_missing_live_prefix_fails_but_oracle_can_audit_missingness(tmp_path):
    from stocker_execution.acquired_candidates import RecordedOpeningSource
    from stocker_execution.ibkr import IbkrError

    class Partial:
        history = None

        async def prefix(self, *args, **kwargs):
            return ()

    async def scenario():
        _, _, session = setup_run(count=1)
        prefix = session.minute_prefix(5)
        due = prefix[-1] + timedelta(minutes=1)
        identity = CandidateIdentity(1, "S1", None, "SMART", "USD", MarketId.US_ALL, "STK")
        store = AcquisitionStore(tmp_path / "partial.sqlite")
        source = RecordedOpeningSource(Partial(), store, "run", lambda: due)
        with pytest.raises(IbkrError, match="MISSING_REQUIRED_OPENING_PREFIX"):
            await source.prefix(identity, prefix, due)
        oracle = RecordedOpeningSource(Partial(), store, "run", lambda: due, "AUDIT")
        assert await oracle.prefix(identity, prefix, due) == ()
        rows = store.details("run", session.session, "requests", 10, 0)
        assert len(rows) == 2 and rows[0]["payload"]["missing_prefix"]

    asyncio.run(scenario())


@pytest.mark.parametrize("sweeps", [(60,), (60, 180, 240)])
def test_contract_checks_do_not_hold_scanner_slots_or_delay_sweeps(tmp_path, sweeps):
    """US 2026-09-11: slow contract checks blocked later scanner snapshots."""

    async def scenario():
        _, instance, session = setup_run(count=10)
        now = [session.opens_at]
        all_scans_received = asyncio.Event()
        expected_requests = 35 * len(sweeps)

        class SlowQualificationBroker(Broker):
            async def acquisition_scan(self, request, audit):
                rows = await super().acquisition_scan(request, audit)
                if len(self.requests) == expected_requests:
                    all_scans_received.set()
                return rows

            async def qualify_discovery_candidate(self, row):
                await all_scans_received.wait()
                return await super().qualify_discovery_candidate(row)

        async def wait(due):
            now[0] = due

        broker = SlowQualificationBroker()
        recipe = ACQUISITION_EXPERIMENT_V1.model_copy(update={"sweep_active_seconds": sweeps})
        store = AcquisitionStore(tmp_path / "nonblocking-scans.sqlite")
        provider = ScannerAcquisition(broker, store, lambda: now[0], wait, recipe)
        pool, errors = await asyncio.wait_for(provider.acquire(instance), timeout=2)
        assert not errors
        assert len(pool) == 6
        assert len(broker.requests) == expected_requests
        assert broker.maximum == recipe.scanner_concurrency
        assert broker.qualifications == 6
        assert store.summary(instance.config.run_id, session.session)["state"] == (
            "SCANNER_ACQUISITION_READY"
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("ending", ["cancel", "deadline"])
def test_pending_contract_checks_cannot_write_after_acquisition_seals(tmp_path, ending):
    async def scenario():
        _, instance, session = setup_run(count=10)
        now = [session.opens_at]
        entered = asyncio.Event()
        release = asyncio.Event()

        class WaitingBroker(Broker):
            pending = 0

            async def qualify_discovery_candidate(self, row):
                self.pending += 1
                entered.set()
                try:
                    await release.wait()
                    return await super().qualify_discovery_candidate(row)
                finally:
                    self.pending -= 1

        async def wait(due):
            now[0] = max(now[0], due)

        broker = WaitingBroker()
        recipe = ACQUISITION_EXPERIMENT_V1.model_copy(update={"sweep_active_seconds": (60,)})
        store = AcquisitionStore(tmp_path / "cancel-qualification.sqlite")
        provider = ScannerAcquisition(broker, store, lambda: now[0], wait, recipe)
        task = asyncio.create_task(provider.acquire(instance))
        await asyncio.wait_for(entered.wait(), timeout=2)
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            now[0] = session.minute_prefix(5)[-1] + timedelta(minutes=1)
            release.set()
            _, errors = await asyncio.wait_for(task, timeout=2)
            assert errors[0]["acquisition_failure"]
        assert broker.pending == 0
        saved = store.session(instance.config.run_id, session.session)
        assert saved["sealed"]
        assert store.pool(instance.config.run_id, session.session) == ()
        before = store.details(instance.config.run_id, session.session, "components", 200, 0)
        release.set()
        await asyncio.sleep(0)
        assert (
            store.details(instance.config.run_id, session.session, "components", 200, 0) == before
        )
        calls = len(broker.requests)
        _, errors = await provider.acquire(instance)
        assert errors[0]["acquisition_failure"] and len(broker.requests) == calls

    asyncio.run(scenario())


def test_acquisition_timeout_has_an_actionable_persisted_reason(tmp_path):
    class TimedOut:
        async def acquire(self, instance):
            raise TimeoutError()

    async def scenario():
        _, instance, session = setup_run(count=1)
        now = [session.opens_at]
        store = CandidateStore(tmp_path / "deadline-reason.sqlite")
        pipeline = CandidatePipeline(store, TimedOut(), Source(), lambda: now[0])
        await drain(pipeline, instance, session, now)
        summary = store.summary(instance.config.run_id, session.session)
        assert summary["state"] == "DEGRADED"
        assert "ACQUISITION_DEADLINE_EXCEEDED" in summary["reason"]
        assert summary["completed_stages"] == 0

    asyncio.run(scenario())
