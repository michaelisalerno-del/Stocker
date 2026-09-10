"""Durable opening-candidate lifecycle at the existing method universe boundary."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from stocker_core.candidate_selection import (
    SESSION_HARD_CANDIDATE_RECIPE,
    CandidateIdentity,
    CandidateRank,
    candidate_value,
    rank_candidates,
    require_aware,
)
from stocker_core.markets import get_market
from stocker_core.runs import RunInstance
from stocker_execution.discovery import DiscoveryRow
from stocker_execution.history import IbkrHistoryCache, IbkrHistoryService
from stocker_execution.ibkr import (
    HISTORICAL_REQUEST_CONCURRENCY,
    HistoricalBar,
    IbkrConnection,
    IbkrHistoricalDataUnavailable,
    IbkrInstrumentUnavailable,
    QualifiedInstrument,
)
from stocker_execution.stage5 import (
    Stage5IneligibleInstrument,
    Stage5Membership,
    Stage5QualificationResult,
    Stage5QualifiedRequest,
)

if TYPE_CHECKING:
    from stocker_execution.runtime import MarketSession

STAGES = SESSION_HARD_CANDIDATE_RECIPE.stages
STATES = ("BROAD_ELIGIBLE", "RANGE5_SELECTED", "RV10_SELECTED", "RV15_SELECTED")
CAPACITY_REASON = "BROAD_OPENING_DATA_CAPACITY_UNRESOLVED"


def instrument(identity: CandidateIdentity) -> QualifiedInstrument:
    return QualifiedInstrument(
        identity.symbol,
        identity.con_id,
        identity.routing_exchange,
        identity.primary_exchange,
        identity.currency,
        identity.security_type,
    )


class UniverseProvider(Protocol):
    async def acquire(
        self, instance: RunInstance
    ) -> tuple[tuple[CandidateIdentity, ...], list[dict[str, Any]]]: ...


class ConfiguredUniverseProvider:
    """Resolve the entire saved population; no scanner or activity ranking is called."""

    def __init__(self, broker: IbkrConnection):
        self.broker = broker

    async def acquire(
        self, instance: RunInstance
    ) -> tuple[tuple[CandidateIdentity, ...], list[dict[str, Any]]]:
        market_id = instance.config.market_id
        assert market_id is not None
        market = get_market(market_id)
        eligible: dict[int, CandidateIdentity] = {}
        rejected: list[dict[str, Any]] = []
        semaphore = asyncio.Semaphore(HISTORICAL_REQUEST_CONCURRENCY)

        async def qualify(reference: Any) -> None:
            async with semaphore:
                reason = ""
                acquisition_failure = False
                try:
                    if reference.security_type != "STK" or reference.currency != market.currency:
                        raise ValueError("WRONG_SECURITY_TYPE_OR_MARKET")
                    resolved = await self.broker.resolve_stock(
                        reference.symbol,
                        exchange=reference.exchange,
                        primary_exchange=reference.primary_exchange,
                        currency=reference.currency,
                    )
                    row = DiscoveryRow(
                        resolved.con_id,
                        resolved.symbol,
                        resolved.exchange,
                        resolved.primary_exchange,
                        resolved.currency,
                        resolved.security_type,
                        0,
                        {},
                    )
                    resolved, stock_type = await self.broker.qualify_discovery_candidate(row)
                    if (
                        resolved.security_type != "STK"
                        or resolved.currency != market.currency
                        or stock_type not in {"COMMON", "CORP", "ADR", "REIT"}
                    ):
                        raise ValueError("WRONG_SECURITY_TYPE_OR_MARKET")
                    eligible[resolved.con_id] = CandidateIdentity(
                        resolved.con_id,
                        resolved.symbol,
                        resolved.primary_exchange,
                        resolved.primary_exchange
                        or reference.primary_exchange
                        or resolved.exchange,
                        resolved.currency,
                        market_id,
                        resolved.security_type,
                        resolved.exchange,
                    )
                except (ValueError, IbkrInstrumentUnavailable) as exc:
                    reason = str(exc)
                except Exception as exc:
                    reason = str(exc)
                    acquisition_failure = True
                if reason:
                    rejected.append(
                        {
                            "reference": reference.model_dump(mode="json"),
                            "reason": reason,
                            "acquisition_failure": acquisition_failure,
                        }
                    )

        # Bounded workers rather than thousands of waiting broker tasks.
        refs = iter(instance.universe.members)

        async def worker() -> None:
            for ref in refs:
                await qualify(ref)
                await asyncio.sleep(0)

        await asyncio.gather(*(worker() for _ in range(HISTORICAL_REQUEST_CONCURRENCY)))
        return tuple(eligible.values()), rejected


class OpeningBarSource:
    """Exact end-time RTH minute requests on the existing IBKR history ingress.

    No TRADES subscriptions, quote lines, prior HV or method features are allocated.
    Concurrent candidate runs share completed/in-flight requests within this service.
    """

    def __init__(self, broker: IbkrConnection, cache: IbkrHistoryCache):
        self.history = IbkrHistoryService(broker, cache)
        self._bars: dict[tuple[int, datetime, tuple[datetime, ...]], tuple[HistoricalBar, ...]] = {}
        self._locks: dict[tuple[int, datetime, tuple[datetime, ...]], asyncio.Lock] = {}

    async def prefix(
        self, identity: CandidateIdentity, expected: tuple[datetime, ...], due: datetime
    ) -> tuple[HistoricalBar, ...]:
        self.prune(due.date() - timedelta(days=1))
        key = identity.con_id, due, expected
        async with self._locks.setdefault(key, asyncio.Lock()):
            if key not in self._bars:
                try:
                    rows = await self.history.fetch_and_store(
                        instrument(identity),
                        bar_size="1 min",
                        duration=f"{int((due - expected[0]).total_seconds())} S",
                        what_to_show="TRADES",
                        regular_trading_hours=True,
                        end_time=due,
                    )
                except IbkrHistoricalDataUnavailable:
                    rows = ()
                self._bars[key] = tuple(b for b in rows if b.timestamp in expected)
            return self._bars[key]

    def prune(self, today: date) -> None:
        self._bars = {k: v for k, v in self._bars.items() if k[1].date() >= today}
        self._locks = {k: v for k, v in self._locks.items() if k[1].date() >= today or v.locked()}


class CandidateStore:
    """Small summary row plus normalized population/stage rows; atomic stage commits."""

    def __init__(self, path: Path, *, initialize: bool = True):
        self.path = path
        if initialize:
            with self.connect() as db:
                db.executescript("""
                CREATE TABLE IF NOT EXISTS opening_candidate_sessions (
                    run_id TEXT NOT NULL, session TEXT NOT NULL, market TEXT NOT NULL,
                    spec_hash TEXT NOT NULL, metadata TEXT NOT NULL, state TEXT NOT NULL,
                    completed_stages INTEGER NOT NULL DEFAULT 0, reason TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL, PRIMARY KEY(run_id, session)
                );
                CREATE TABLE IF NOT EXISTS opening_candidate_population (
                    run_id TEXT NOT NULL, session TEXT NOT NULL, con_id INTEGER NOT NULL,
                    identity TEXT NOT NULL, PRIMARY KEY(run_id,session,con_id)
                );
                CREATE TABLE IF NOT EXISTS opening_candidate_sources (
                    run_id TEXT NOT NULL, session TEXT NOT NULL, source_index INTEGER NOT NULL,
                    reference TEXT NOT NULL, PRIMARY KEY(run_id,session,source_index)
                );
                CREATE TABLE IF NOT EXISTS opening_candidate_rejections (
                    run_id TEXT NOT NULL, session TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS opening_candidate_stages (
                    run_id TEXT NOT NULL, session TEXT NOT NULL, stage INTEGER NOT NULL,
                    con_id INTEGER NOT NULL, score_name TEXT NOT NULL, feature_value REAL,
                    rank INTEGER NOT NULL, selected INTEGER NOT NULL, missing_reason TEXT NOT NULL,
                    due_at TEXT NOT NULL, selected_at TEXT NOT NULL, input_bars TEXT NOT NULL,
                    PRIMARY KEY(run_id,session,stage,con_id)
                );
                CREATE INDEX IF NOT EXISTS opening_candidate_selected
                    ON opening_candidate_stages(run_id,session,stage,selected);
                """)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def summary(self, run_id: str, session: date) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with self.connect() as db:
            if not db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='opening_candidate_sessions'"
            ).fetchone():
                return None
            row = db.execute(
                "SELECT * FROM opening_candidate_sessions WHERE run_id=? AND session=?",
                (run_id, session.isoformat()),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result.update(json.loads(result.pop("metadata")))
            result["broad_eligible"] = db.execute(
                "SELECT COUNT(*) FROM opening_candidate_population WHERE run_id=? AND session=?",
                (run_id, session.isoformat()),
            ).fetchone()[0]
            counts = dict(
                db.execute(
                    "SELECT stage, SUM(selected) FROM opening_candidate_stages "
                    "WHERE run_id=? AND session=? GROUP BY stage",
                    (run_id, session.isoformat()),
                ).fetchall()
            )
            result["stages"] = [
                dict(
                    stage_id=s.stage_id,
                    score_name=s.score_name,
                    minutes=s.minutes,
                    capacity=s.capacity,
                    selected=counts.get(i),
                    state=STATES[i + 1],
                )
                for i, s in enumerate(STAGES)
            ]
            result["ready"] = result["state"] == "SESSION_HARD_ACTIVE"
            result["watchlist_size"] = counts.get(len(STAGES) - 1, 0)
            return result

    def begin(self, instance: RunInstance, market: MarketSession, now: datetime) -> None:
        run = instance.config
        assert run.market_id is not None and run.method_spec is not None
        metadata = {
            "recipe": run.method_spec["candidate_selection"],
            "source": str(run.universe_source),
            "source_population_count": len(instance.universe.members),
            "calendar": run.session.calendar if run.session else None,
            "timezone": run.session.timezone if run.session else None,
            "opens_at": market.opens_at.isoformat() if market.opens_at else None,
            "closes_at": market.closes_at.isoformat() if market.closes_at else None,
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO opening_candidate_sessions VALUES (?,?,?,?,?,'DISCOVERY',0,'',?)",
                (
                    run.run_id,
                    market.session.isoformat(),
                    run.market_id.value,
                    run.method_spec_hash,
                    json.dumps(metadata),
                    now.isoformat(),
                ),
            )
            db.executemany(
                "INSERT INTO opening_candidate_sources VALUES (?,?,?,?)",
                (
                    (run.run_id, market.session.isoformat(), i, r.model_dump_json())
                    for i, r in enumerate(instance.universe.members)
                ),
            )

    def population(
        self, run_id: str, session: date, survivors_of: int | None = None
    ) -> tuple[CandidateIdentity, ...]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT identity FROM opening_candidate_population p WHERE run_id=? AND session=? "
                "AND (? IS NULL OR con_id IN (SELECT con_id FROM opening_candidate_stages "
                "WHERE run_id=p.run_id AND session=p.session AND stage=? AND selected=1))",
                (run_id, session.isoformat(), survivors_of, survivors_of),
            ).fetchall()
        return tuple(CandidateIdentity(**json.loads(r[0])) for r in rows)

    def save_population(
        self,
        run_id: str,
        session: date,
        identities: Sequence[CandidateIdentity],
        rejected: list[dict[str, Any]],
        now: datetime,
        failure_reason: str = "",
    ) -> None:
        with self.connect() as db:
            db.executemany(
                "INSERT INTO opening_candidate_population VALUES (?,?,?,?)",
                (
                    (run_id, session.isoformat(), i.con_id, json.dumps(asdict(i)))
                    for i in identities
                ),
            )
            db.executemany(
                "INSERT INTO opening_candidate_rejections VALUES (?,?,?)",
                ((run_id, session.isoformat(), json.dumps(r)) for r in rejected),
            )
            db.execute(
                "UPDATE opening_candidate_sessions SET state=?,reason=?,updated_at=? "
                "WHERE run_id=? AND session=?",
                ("DEGRADED" if failure_reason else "BROAD_ELIGIBLE", failure_reason,
                 now.isoformat(), run_id, session.isoformat()),
            )

    def fail(self, run_id: str, session: date, reason: str, now: datetime) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE opening_candidate_sessions SET "
                "state='DEGRADED',reason=?,updated_at=? WHERE run_id=? AND session=?",
                (reason, now.isoformat(), run_id, session.isoformat()),
            )

    def commit_stage(
        self,
        run_id: str,
        session: date,
        index: int,
        ranked: Sequence[CandidateRank],
        bars: dict[int, tuple[HistoricalBar, ...]],
        due: datetime,
        now: datetime,
        failure_reason: str = "",
    ) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT completed_stages, state FROM opening_candidate_sessions WHERE "
                "run_id=? AND session=?",
                (run_id, session.isoformat()),
            ).fetchone()
            if row is None or row[0] != index or row[1] != STATES[index]:
                raise ValueError("Candidate stage cannot be replaced or run out of order")
            db.executemany(
                "INSERT INTO opening_candidate_stages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    (
                        run_id,
                        session.isoformat(),
                        index,
                        r.identity.con_id,
                        r.score_name,
                        r.value,
                        r.rank,
                        int(r.selected),
                        r.missing_reason,
                        due.isoformat(),
                        now.isoformat(),
                        json.dumps(
                            [asdict(b) for b in bars.get(r.identity.con_id, ())], default=str
                        ),
                    )
                    for r in ranked
                ),
            )
            db.execute(
                "UPDATE opening_candidate_sessions SET "
                "state=?,completed_stages=?,updated_at=?,reason=? WHERE run_id=? AND "
                "session=?",
                (
                    "DEGRADED" if failure_reason else STATES[index + 1],
                    index + 1,
                    now.isoformat(),
                    failure_reason,
                    run_id,
                    session.isoformat(),
                ),
            )

    def activate(self, run_id: str, session: date, now: datetime) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE opening_candidate_sessions SET state='SESSION_HARD_ACTIVE',updated_at=? "
                "WHERE run_id=? AND session=? AND state='RV15_SELECTED'",
                (now.isoformat(), run_id, session.isoformat()),
            )

    def details(self, run_id: str, session: date, limit: int, offset: int) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT p.identity,s.* FROM opening_candidate_population p LEFT JOIN "
                "opening_candidate_stages s "
                "ON p.run_id=s.run_id AND p.session=s.session AND p.con_id=s.con_id "
                "WHERE p.run_id=? AND p.session=? ORDER BY p.con_id,s.stage LIMIT ? OFFSET ?",
                (run_id, session.isoformat(), limit, offset),
            ).fetchall()
        return [dict(r) | {"identity": json.loads(r["identity"])} for r in rows]


class CandidatePipeline:
    def __init__(
        self,
        store: CandidateStore,
        provider: UniverseProvider,
        source: OpeningBarSource,
        clock: Callable[[], datetime],
    ):
        self.store, self.provider, self.source, self.clock = store, provider, source, clock
        self.tasks: dict[tuple[str, date], asyncio.Task[None]] = {}
        self.seen: set[tuple[str, date]] = set()
        self.delivered: set[tuple[str, date]] = set()

    async def qualify(self, runs: Sequence[RunInstance]) -> Stage5QualificationResult:
        from stocker_execution.runtime import ExchangeSessionResolver

        results = []
        for instance in runs:
            now = require_aware(self.clock())
            market = ExchangeSessionResolver().resolve(instance.config, now)
            result = await self.advance(instance, market, now)
            results.append(result or self.result(instance, market.session))
        return Stage5QualificationResult(
            tuple(r for result in results for r in result.requests),
            tuple(r for result in results for r in result.ineligible),
        )

    def result(self, instance: RunInstance, session: date) -> Stage5QualificationResult:
        run = instance.config
        summary = self.store.summary(run.run_id, session)
        membership = Stage5Membership(run.run_id, instance.universe.universe_id)
        if summary and summary["ready"]:
            identities = self.store.population(run.run_id, session, len(STAGES) - 1)
            return Stage5QualificationResult(
                tuple(Stage5QualifiedRequest(instrument(i), (membership,)) for i in identities), ()
            )
        return Stage5QualificationResult(
            (),
            (
                Stage5IneligibleInstrument(
                    SESSION_HARD_CANDIDATE_RECIPE.recipe_id,
                    (membership,),
                    summary["reason"] or summary["state"]
                    if summary
                    else "CANDIDATE_SELECTION_PENDING",
                ),
            ),
        )

    def ready(self, run_id: str, session: date) -> bool:
        row = self.store.summary(run_id, session)
        return bool(row and row["ready"])

    async def stop(self) -> None:
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()

    async def advance(
        self, instance: RunInstance, market: MarketSession, now: datetime
    ) -> Stage5QualificationResult | None:
        run = instance.config
        key = run.run_id, market.session
        for old_key, old_task in tuple(self.tasks.items()):
            if old_key[0] == run.run_id and (old_key != key or not run.enabled):
                old_task.cancel()
                await asyncio.gather(old_task, return_exceptions=True)
                del self.tasks[old_key]
        if not run.enabled or market.opens_at is None or market.closes_at is None:
            return None
        task = self.tasks.get(key)
        if task is not None:
            if not task.done():
                return None
            del self.tasks[key]
            if not task.cancelled() and task.exception() is not None:
                self.store.fail(*key, f"{CAPACITY_REASON}: {task.exception()}", now)
            return self.result(instance, market.session)
        summary = self.store.summary(*key)
        if summary is None:
            await asyncio.to_thread(self.store.begin, instance, market, now)
            summary = self.store.summary(*key)
        assert summary is not None
        if summary["market"] != run.market_id or summary["spec_hash"] != run.method_spec_hash:
            raise ValueError("Candidate session identity/specification mismatch")
        if summary["state"] == "RV15_SELECTED":
            self.store.activate(*key, now)
            summary = self.store.summary(*key)
            assert summary is not None
        index = int(summary["completed_stages"])
        if summary["ready"] or summary["state"] == "DEGRADED":
            if key not in self.delivered:
                self.delivered.add(key)
                return self.result(instance, market.session)
            return None
        prefix = market.minute_prefix(STAGES[index].minutes)
        due = prefix[-1] + timedelta(minutes=1)
        if key not in self.seen:
            self.seen.add(key)
            if now >= due:
                self.store.fail(*key, "CANDIDATE_SELECTION_WINDOW_MISSED", now)
                return self.result(instance, market.session)
        if summary["state"] == "DISCOVERY":
            if not instance.universe.members:
                self.store.fail(*key, "BROAD_UNIVERSE_UNAVAILABLE", now)
                return self.result(instance, market.session)
            self.tasks[key] = asyncio.create_task(self._acquire(instance, market, due))
        elif now >= due:
            assert run.method_spec is not None
            final_limit = dict(market.checkpoint_times())[
                run.method_spec["qualification"]["checkpoints"][0]
            ]
            limit = (
                market.minute_prefix(STAGES[index + 1].minutes)[-1] + timedelta(minutes=1)
                if index + 1 < len(STAGES)
                else final_limit
            )
            if now >= limit:
                self.store.fail(*key, "CANDIDATE_SELECTION_WINDOW_MISSED", now)
                return self.result(instance, market.session)
            self.tasks[key] = asyncio.create_task(
                self._stage(instance, market, index, prefix, due, limit)
            )
        return None

    async def _acquire(self, instance: RunInstance, market: MarketSession, due: datetime) -> None:
        key = instance.config.run_id, market.session
        try:
            async with asyncio.timeout(max(0, (due - self.clock()).total_seconds())):
                identities, rejected = await self.provider.acquire(instance)
            failure_reason = ""
            if any(r.get("acquisition_failure") for r in rejected):
                failure_reason = f"{CAPACITY_REASON}: Broad source was not fully qualified"
            elif not identities or self.clock() >= due:
                failure_reason = f"{CAPACITY_REASON}: Broad preparation missed its opening deadline"
            await asyncio.to_thread(
                self.store.save_population, *key, identities, rejected, self.clock(), failure_reason
            )
        except asyncio.CancelledError:
            self.store.fail(*key, "CANDIDATE_SELECTION_INTERRUPTED", self.clock())
            raise
        except Exception as exc:
            self.store.fail(*key, f"{CAPACITY_REASON}: {exc}", self.clock())

    async def _stage(
        self,
        instance: RunInstance,
        market: MarketSession,
        index: int,
        prefix: tuple[datetime, ...],
        due: datetime,
        limit: datetime,
    ) -> None:
        key = instance.config.run_id, market.session
        try:
            identities = self.store.population(*key, index - 1 if index else None)
            collected: dict[int, tuple[HistoricalBar, ...]] = {}
            failures: list[str] = []
            remaining = iter(identities)

            async def worker() -> None:
                for identity in remaining:
                    try:
                        collected[identity.con_id] = await self.source.prefix(identity, prefix, due)
                    except Exception as exc:
                        # A successful partial prefix is missing-last; transport failures
                        # are acquisition failures, not zero movement or hash-filled slots.
                        failures.append(f"{identity.con_id}: {exc}")
                        collected[identity.con_id] = ()
                    await asyncio.sleep(0)

            async with asyncio.timeout(max(0, (limit - self.clock()).total_seconds())):
                await asyncio.gather(*(worker() for _ in range(HISTORICAL_REQUEST_CONCURRENCY)))
            stage = STAGES[index]
            values = {
                i.con_id: candidate_value(
                    stage, collected[i.con_id], expected_prefix=prefix, as_of=due
                )
                for i in identities
            }
            assert instance.config.market_id is not None
            ranked = rank_candidates(stage, identities, values, market=instance.config.market_id)
            failure_reason = (
                f"{CAPACITY_REASON}: " + ("; ".join(failures[:3]) or "stage deadline exceeded")
                if failures or self.clock() >= limit
                else ""
            )
            await asyncio.to_thread(
                self.store.commit_stage,
                *key,
                index,
                ranked,
                collected,
                due,
                self.clock(),
                failure_reason,
            )
            if not failure_reason and index == len(STAGES) - 1:
                self.store.activate(*key, self.clock())
        except asyncio.CancelledError:
            self.store.fail(*key, "CANDIDATE_SELECTION_INTERRUPTED", self.clock())
            raise
        except Exception as exc:
            self.store.fail(*key, f"{CAPACITY_REASON}: {exc}", self.clock())
