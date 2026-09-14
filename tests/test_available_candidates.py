"""Missing opening prints must only reject that stock in the new PAPER policy."""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from stocker_execution.acquired_candidates import AcquiredCandidates
from stocker_execution.candidate_pipeline import CandidateStore
from stocker_execution.history import IbkrHistoryCache
from stocker_execution.ibkr import (
    HistoricalBar,
    IbkrConnection,
    IbkrError,
    IbkrHistoricalDataUnavailable,
)
from test_candidate_pipeline import Provider, drain, setup_run


def test_oracle_rejects_invalid_scores_under_the_new_policy_but_retains_old_ranking():
    from stocker_core.candidate_selection import CandidateMissingPolicy
    from stocker_execution.candidate_oracle import reconstruct

    async def scenario():
        _, instance, session = setup_run(count=2)
        identities, _ = await Provider().acquire(instance)
        prefix = session.minute_prefix(15)
        bars = {
            1: tuple(HistoricalBar(t, 10, 9, 8, 10, 1) for t in prefix),
            2: tuple(HistoricalBar(t, 10, 10, 10, 10, 1) for t in prefix),
        }
        old = reconstruct(identities, bars, prefix, instance.config.market_id)
        new = reconstruct(
            identities,
            bars,
            prefix,
            instance.config.market_id,
            CandidateMissingPolicy.REJECT_UNAVAILABLE,
        )
        assert new[0][1].missing_reason == "INVALID_OHLC"
        assert new[0][0].value == 0 and new[0][0].selected
        assert all({r.identity.con_id for r in stage if r.selected} == {2} for stage in new)
        assert all({r.identity.con_id for r in stage if r.selected} == {1, 2} for stage in old)

    asyncio.run(scenario())


def test_v9_hashes_and_trading_rules_are_preserved_and_v10_is_paper_only():
    from stocker_core.methods import (
        SESSION_HARD,
        SESSION_HARD_ACQUISITION_V9,
        content_hash,
        validate_run_method,
    )
    from stocker_core.runs import Environment
    from stocker_dashboard.universe_runs import UniverseRunBuilder

    expected = json.loads(
        (Path(__file__).parent / "fixtures/acquisition_v9_spec_hashes.json").read_text()
    )
    for market, digest in expected.items():
        old = SESSION_HARD_ACQUISITION_V9.specification(market)
        new = SESSION_HARD.specification(market)
        assert content_hash(old) == digest
        assert {k for k in old if old[k] != new[k]} == {"method_version", "candidate_selection"}
        assert content_hash(new) != digest
        assert new["candidate_selection"]["stages"] == old["candidate_selection"]["stages"]
    config, instance, _ = setup_run(count=2)
    _, old = UniverseRunBuilder().add(
        config,
        market_id=instance.config.market_id,
        strategy_id=SESSION_HARD_ACQUISITION_V9.method_id,
        strategy_version=SESSION_HARD_ACQUISITION_V9.version,
        environment=Environment.PAPER,
        historical_reproduction=True,
    )
    validate_run_method(old)
    validate_run_method(instance.config)
    with pytest.raises(ValueError, match="PAPER-only"):
        UniverseRunBuilder().add(
            config,
            market_id=instance.config.market_id,
            strategy_id=SESSION_HARD.method_id,
            strategy_version=SESSION_HARD.version,
            environment=Environment.LIVE,
        )


def test_saved_v9_still_blocks_incomplete_opening_prefixes(tmp_path):
    from stocker_core.methods import SESSION_HARD_ACQUISITION_V9
    from stocker_core.runs import Environment
    from stocker_dashboard.universe_runs import UniverseRunBuilder

    async def scenario():
        config, instance, session = setup_run(count=1)
        _, old = UniverseRunBuilder().add(
            config,
            market_id=instance.config.market_id,
            strategy_id=SESSION_HARD_ACQUISITION_V9.method_id,
            strategy_version=SESSION_HARD_ACQUISITION_V9.version,
            environment=Environment.PAPER,
            historical_reproduction=True,
        )
        instance = replace(instance, config=old, universe=old.universe_snapshot)
        now = [session.opens_at - timedelta(minutes=1)]

        class Broker(IbkrConnection):
            def __init__(self):
                pass

            async def historical_bars(self, *args, **kwargs):
                raise IbkrHistoricalDataUnavailable("no data")

        service = AcquiredCandidates(
            Broker(),
            IbkrHistoryCache(tmp_path / "bars.sqlite"),
            CandidateStore(tmp_path / "state.sqlite"),
            lambda: now[0],
        )
        service.provider = Provider()
        pipeline = service.pipeline(instance)
        await drain(pipeline, instance, session, now)
        now[0] = session.opens_at + timedelta(minutes=5)
        await drain(pipeline, instance, session, now)
        summary = pipeline.store.summary(old.run_id, session.session)
        assert summary["state"] == "DEGRADED"
        assert "MISSING_REQUIRED_OPENING_PREFIX" in summary["reason"]
        await service.stop()

    asyncio.run(scenario())


def test_missing_stock_prefixes_do_not_block_valid_candidates_or_replenish(tmp_path):
    async def scenario():
        _, instance, session = setup_run(count=4)
        now = [session.opens_at - timedelta(minutes=1)]

        class Broker(IbkrConnection):
            def __init__(self):
                self.calls = []

            async def historical_bars(self, stock, **kwargs):
                minutes = int(kwargs["duration"].split()[0]) // 60
                self.calls.append((stock.con_id, minutes))
                if stock.con_id == 2 or (stock.con_id == 3 and minutes >= 10):
                    raise IbkrHistoricalDataUnavailable("HMDS query returned no data")
                prefix = session.minute_prefix(minutes)
                if stock.con_id == 1:
                    prefix = prefix[2:]
                return tuple(HistoricalBar(t, 10, 11, 9, 10, 1) for t in prefix)

        broker = Broker()
        service = AcquiredCandidates(
            broker,
            IbkrHistoryCache(tmp_path / "bars.sqlite"),
            CandidateStore(tmp_path / "state.sqlite"),
            lambda: now[0],
        )
        service.provider = Provider()
        pipeline = service.pipeline(instance)
        await drain(pipeline, instance, session, now)
        key = instance.config.run_id, session.session
        for index, minute in enumerate((5, 10, 15)):
            now[0] = session.opens_at + timedelta(minutes=minute)
            await drain(pipeline, instance, session, now)
            summary = pipeline.store.summary(*key)
            assert summary["state"] != "DEGRADED", summary["reason"]
            selected = {i.con_id for i in pipeline.store.population(*key, index)}
            assert selected == ({3, 4} if index == 0 else {4})
            assert summary["stages"][index]["unavailable_rejected"] == (2, 1, 0)[index]
            if index == 0:
                await service.stop()
                service = AcquiredCandidates(
                    broker,
                    IbkrHistoryCache(tmp_path / "bars.sqlite"),
                    CandidateStore(tmp_path / "state.sqlite"),
                    lambda: now[0],
                )
                service.provider = Provider()
                pipeline = service.pipeline(instance)
                await pipeline.advance(instance, session, now[0])
        assert {
            r.instrument.con_id for r in pipeline.result(instance, session.session).requests
        } == {4}
        assert set(broker.calls) == {(1, 5), (2, 5), (3, 5), (4, 5), (3, 10), (4, 10), (4, 15)}
        with pipeline.store.connect() as db:
            rows = db.execute(
                "SELECT con_id,selected,missing_reason,input_bars FROM opening_candidate_stages "
                "WHERE stage=0 ORDER BY con_id"
            ).fetchall()
        assert [(r[0], r[1], r[2]) for r in rows[:2]] == [
            (1, 0, "MISSING_BAR"),
            (2, 0, "MISSING_BAR"),
        ]
        assert "timestamp" in rows[0][3]  # Partial evidence is retained, not replaced by [].
        await service.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["no_data", "permission", "connection", "pacing", "deadline"])
def test_empty_population_and_broker_failures_still_block_the_new_policy(tmp_path, failure):
    async def scenario():
        _, instance, session = setup_run(count=2)
        now = [session.opens_at - timedelta(minutes=1)]

        class Broker(IbkrConnection):
            def __init__(self):
                pass

            async def historical_bars(self, stock, **kwargs):
                if failure == "no_data":
                    raise IbkrHistoricalDataUnavailable("no bars")
                if failure == "deadline":
                    now[0] = session.opens_at + timedelta(minutes=10)
                    return tuple(
                        HistoricalBar(t, 10, 11, 9, 10, 1) for t in session.minute_prefix(5)
                    )
                raise IbkrError(
                    {
                        "permission": "No market data permissions",
                        "connection": "disconnected",
                        "pacing": "historical pacing violation",
                    }[failure]
                )

        service = AcquiredCandidates(
            Broker(),
            IbkrHistoryCache(tmp_path / "bars.sqlite"),
            CandidateStore(tmp_path / "state.sqlite"),
            lambda: now[0],
        )
        service.provider = Provider()
        pipeline = service.pipeline(instance)
        await drain(pipeline, instance, session, now)
        now[0] = session.opens_at + timedelta(minutes=5)
        await drain(pipeline, instance, session, now)
        assert (
            pipeline.store.summary(instance.config.run_id, session.session)["state"] == "DEGRADED"
        )
        assert not pipeline.result(instance, session.session).requests
        if failure == "no_data":
            assert (
                "NO_VALID_OPENING_CANDIDATES"
                in pipeline.store.summary(instance.config.run_id, session.session)["reason"]
            )
        await service.stop()

    asyncio.run(scenario())
