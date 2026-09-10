"""Delayed acquisition research. This module cannot update candidate or strategy tables."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta
from math import isfinite
from typing import Any

import structlog

from stocker_core.candidate_selection import (
    SESSION_HARD_CANDIDATE_RECIPE,
    CandidateIdentity,
    CandidateRank,
    CandidateValue,
    candidate_value,
    rank_candidates,
)
from stocker_core.markets import MarketId
from stocker_core.runs import RunInstance
from stocker_core.universes import InstrumentReference
from stocker_execution.acquisition_store import AcquisitionStore
from stocker_execution.candidate_pipeline import (
    ConfiguredUniverseProvider,
    OpeningBarSource,
    instrument,
)
from stocker_execution.ibkr import HistoricalBar, IbkrConnection, IbkrHistoricalDataUnavailable

STAGES = SESSION_HARD_CANDIDATE_RECIPE.stages


def decoded_bars(payload: str | None) -> tuple[HistoricalBar, ...]:
    return tuple(
        HistoricalBar(**(row | {"timestamp": datetime.fromisoformat(row["timestamp"])}))
        for row in json.loads(payload or "[]")
    )


def reconstruct(
    population: Sequence[CandidateIdentity],
    bars: dict[int, tuple[HistoricalBar, ...]],
    prefix: tuple[datetime, ...],
    market: MarketId,
) -> list[tuple[CandidateRank, ...]]:
    active = tuple(population)
    rankings = []
    for stage in STAGES:
        required = prefix[: stage.minutes]
        values = {
            i.con_id: candidate_value(
                stage,
                bars[i.con_id],
                expected_prefix=required,
                as_of=required[-1] + timedelta(minutes=1),
            )
            for i in active
        }
        ranked = rank_candidates(stage, active, values, market=market)
        rankings.append(ranked)
        active = tuple(r.identity for r in ranked if r.selected)
    return rankings


def recall_metrics(
    rankings: Sequence[Sequence[CandidateRank]],
    acquired: set[int],
    eligible_union: set[int],
    hits: Sequence[dict[str, Any]],
    components: dict[str, dict[str, Any]],
    shadows: dict[str, dict[str, Any]],
    cutoff: datetime | None = None,
) -> dict[str, Any]:
    by_identity: dict[int, set[str]] = {}
    first_sweep: dict[int, int] = {}
    best_rank: dict[int, int] = {}
    for hit in hits:
        if hit.get("component_status", "COMPLETE") != "COMPLETE":
            continue
        if cutoff is not None and (
            not hit.get("observed_at") or datetime.fromisoformat(hit["observed_at"]) >= cutoff
        ):
            continue
        con_id = hit["con_id"]
        if con_id not in eligible_union:
            continue
        by_identity.setdefault(con_id, set()).add(hit["component"])
        first_sweep[con_id] = min(first_sweep.get(con_id, hit["sweep"]), hit["sweep"])
        best_rank[con_id] = min(best_rank.get(con_id, hit["scanner_rank"]), hit["scanner_rank"])
    recipes = {"ACTIVE": acquired}
    for name, mask in shadows.items():
        allowed = {
            key
            for key, c in components.items()
            if ("families" not in mask or c["family"] in mask["families"])
            and ("cap_slices" not in mask or c["cap_slice"] in mask["cap_slices"])
            and c["cap_slice"] not in mask.get("exclude_cap_slices", ())
        }
        recipes[name] = {con_id for con_id, seen in by_identity.items() if seen & allowed}
    metrics: dict[str, Any] = {
        "label": "AUDIT_ONLY",
        "recipes": {},
        "targets": [],
        "component_contributions": {},
    }
    for name, pool in recipes.items():
        metrics["recipes"][name] = {}
        for stage, ranked in zip(STAGES, rankings, strict=True):
            selected = [r for r in ranked if r.selected]
            available = [r for r in selected if r.value is not None]
            captured = sum(r.identity.con_id in pool for r in available)
            metrics["recipes"][name][stage.stage_id] = {
                "denominator": len(available),
                "captured": captured,
                "missing_selected": len(selected) - len(available),
                "recall": captured / len(available) if available else None,
            }
    for stage, ranked in zip(STAGES, rankings, strict=True):
        for row in ranked:
            if not row.selected:
                continue
            con_id = row.identity.con_id
            seen = sorted(by_identity.get(con_id, ()))
            metrics["targets"].append(
                {
                    "stage": stage.stage_id,
                    "con_id": con_id,
                    "symbol": row.identity.symbol,
                    "oracle_rank": row.rank,
                    "oracle_value": row.value,
                    "missing_reason": row.missing_reason,
                    "captured": con_id in acquired,
                    "components": seen,
                    "component_count": len(seen),
                    "first_sweep": first_sweep.get(con_id),
                    "best_scanner_rank": best_rank.get(con_id),
                }
            )
            if row.value is None:
                continue
            for component in seen:
                key = stage.stage_id + ":" + component
                contribution = metrics["component_contributions"].setdefault(
                    key, {"captured": 0, "unique": 0}
                )
                contribution["captured"] += 1
                contribution["unique"] += len(seen) == 1
    targets = [r for r in rankings[0] if r.selected and r.value is not None]
    metrics["range_rank_buckets"] = {
        f"{low}-{high}": {
            "available": sum(low <= r.rank <= high for r in targets),
            "captured": sum(
                low <= r.rank <= high and r.identity.con_id in acquired for r in targets
            ),
        }
        for low, high in ((1, 25), (26, 50), (51, 100), (101, 150), (151, 200), (201, 250))
    }
    return metrics


def transport_parity(
    population: Sequence[CandidateIdentity],
    bars: dict[int, tuple[HistoricalBar, ...]],
    five_bars: dict[int, tuple[HistoricalBar, ...]],
    prefix: tuple[datetime, ...],
    market: MarketId,
) -> dict[str, Any]:
    stage = STAGES[0]
    exact = {
        i.con_id: candidate_value(
            stage,
            bars[i.con_id],
            expected_prefix=prefix[:5],
            as_of=prefix[4] + timedelta(minutes=1),
        )
        for i in population
    }
    transported = {}
    differences = []
    comparable = 0
    for i in population:
        rows = five_bars.get(i.con_id, ())
        value = None
        if (
            len(rows) == 1
            and rows[0].timestamp == prefix[0]
            and prefix[4] - prefix[0] == timedelta(minutes=4)
            and all(isfinite(x) and x > 0 for x in (rows[0].open, rows[0].high, rows[0].low))
        ):
            value = (rows[0].high - rows[0].low) / rows[0].open
        transported[i.con_id] = CandidateValue(value, "MISSING_BAR" if value is None else "")
        exact_value = exact[i.con_id].value
        if value is not None and exact_value is not None:
            comparable += 1
            differences.append(abs(value - exact_value))
    a = rank_candidates(stage, population, exact, market=market)
    b = rank_candidates(stage, population, transported, market=market)
    return {
        "label": "AUDIT_ONLY_TRANSPORT_EXPERIMENT",
        "population": len(population),
        "comparable": comparable,
        "exact_scores": sum(d == 0 for d in differences),
        "max_absolute_difference": max(differences, default=None),
        "same_ordering": [r.identity.con_id for r in a] == [r.identity.con_id for r in b],
        "same_top250": [r.identity.con_id for r in a if r.selected]
        == [r.identity.con_id for r in b if r.selected],
        "production_transport_changed": False,
    }


class OracleAudit:
    """One background instrument per step, on the existing bounded IBKR ingress."""

    def __init__(
        self,
        broker: IbkrConnection,
        store: AcquisitionStore,
        source: OpeningBarSource,
        clock: Callable[[], datetime],
    ):
        self.broker, self.store, self.source, self.clock = broker, store, source, clock
        self.task: asyncio.Task[None] | None = None

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None

    async def tick(self, instance: RunInstance, allowed: bool) -> None:
        if not allowed:
            if self.task and not self.task.done():
                self.task.cancel()
            return
        if self.task is not None:
            if not self.task.done():
                return
            if not self.task.cancelled() and self.task.exception() is not None:
                structlog.get_logger(__name__).warning(
                    "acquisition_oracle_failed", reason=str(self.task.exception())
                )
            self.task = None
        if (
            self.broker.audit_history_lock.locked()
            or self.broker.resource_status().pending_historical_work
        ):
            return
        # Polling never waits for SQLite or history work.
        self.task = asyncio.create_task(self._next_step(instance))

    def _pending_session(self, run_id: str) -> dict[str, Any] | None:
        with self.store.connect() as db:
            row = db.execute(
                "SELECT session,metadata FROM acquisition_sessions WHERE run_id=? "
                "AND audit_state IN ('PENDING','RUNNING') ORDER BY session LIMIT 1",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    async def _next_step(self, instance: RunInstance) -> None:
        row = await asyncio.to_thread(self._pending_session, instance.config.run_id)
        if row is None:
            return
        metadata = json.loads(row["metadata"])
        if not metadata["oracle_after"] or self.clock() < datetime.fromisoformat(
            metadata["oracle_after"]
        ):
            return
        await self.step(instance, date.fromisoformat(row["session"]), metadata)

    async def step(self, instance: RunInstance, session: date, metadata: dict[str, Any]) -> None:
        key = instance.config.run_id, session
        try:
            async with self.broker.audit_history_lock:
                if self.broker.resource_status().pending_historical_work:
                    return
                await asyncio.to_thread(self.store.audit_state, *key, "RUNNING")
                row = await asyncio.to_thread(self.store.next_oracle_row, *key)
                if row is not None:
                    index = row["source_index"]
                    identity = (
                        CandidateIdentity(**json.loads(row["identity"]))
                        if row["identity"]
                        else None
                    )
                    try:
                        if identity is None:
                            ref = InstrumentReference.model_validate_json(row["reference"])
                            one = replace(
                                instance,
                                universe=instance.universe.model_copy(update={"members": (ref,)}),
                            )
                            identities, rejected = await ConfiguredUniverseProvider(
                                self.broker
                            ).acquire(one)
                            if any(r.get("acquisition_failure") for r in rejected):
                                raise ValueError("AUDIT_CONTRACT_REQUEST_FAILED")
                            if not identities:
                                await asyncio.to_thread(
                                    self.store.audit_progress,
                                    *key,
                                    index,
                                    None,
                                    "INELIGIBLE",
                                    None,
                                    str(rejected),
                                )
                                return
                            identity = identities[0]
                        prefix = tuple(datetime.fromisoformat(t) for t in metadata["prefix"])
                        bars = await self.source.prefix(
                            identity, prefix, prefix[-1] + timedelta(minutes=1)
                        )
                        five = None
                        if metadata["recipe"].get("transport_parity"):
                            try:
                                five = await self.source.history.fetch_and_store(
                                    instrument(identity),
                                    bar_size="5 mins",
                                    duration="300 S",
                                    what_to_show="TRADES",
                                    regular_trading_hours=True,
                                    end_time=prefix[4] + timedelta(minutes=1),
                                )
                                five = tuple(b for b in five if b.timestamp == prefix[0])
                            except IbkrHistoricalDataUnavailable:
                                five = ()
                        await asyncio.to_thread(
                            self.store.audit_progress,
                            *key,
                            index,
                            identity,
                            "ELIGIBLE",
                            bars,
                            "",
                            five,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        await asyncio.to_thread(
                            self.store.audit_progress,
                            *key,
                            index,
                            identity,
                            "ERROR",
                            None,
                            str(exc),
                        )
                    return
                rows = await asyncio.to_thread(self.store.oracle_rows, *key)
                if any(r["eligibility"] == "ERROR" for r in rows):
                    await asyncio.to_thread(self.store.audit_state, *key, "INCOMPLETE")
                    return
                await asyncio.to_thread(self.finish, *key, metadata, rows)
        except asyncio.CancelledError:
            # RUNNING is resumable. Cancellation must not wait for a SQLite writer.
            raise
        except Exception:
            await asyncio.to_thread(self.store.audit_state, *key, "INCOMPLETE")

    def finish(
        self, run_id: str, session: date, metadata: dict[str, Any], rows: list[dict[str, Any]]
    ) -> None:
        population = tuple(
            {
                json.loads(r["identity"])["con_id"]: CandidateIdentity(**json.loads(r["identity"]))
                for r in rows
                if r["identity"]
            }.values()
        )
        bars = {
            json.loads(r["identity"])["con_id"]: decoded_bars(r["bars"])
            for r in rows
            if r["identity"]
        }
        prefix = tuple(datetime.fromisoformat(t) for t in metadata["prefix"])
        rankings = reconstruct(population, bars, prefix, MarketId(metadata["market"]))
        with self.store.connect() as db:
            hits = [
                dict(r)
                for r in db.execute(
                    "SELECT h.*,c.status AS component_status FROM acquisition_hits h "
                    "JOIN acquisition_components c USING(run_id,session,sweep,component) "
                    "WHERE h.run_id=? AND h.session=?",
                    (run_id, str(session)),
                )
            ]
            components = {
                r["component"]: json.loads(r["request"])
                for r in db.execute(
                    "SELECT * FROM acquisition_components WHERE run_id=? AND session=?",
                    (run_id, str(session)),
                )
            }
            pool = list(
                db.execute(
                    "SELECT con_id,selected FROM acquisition_pool WHERE run_id=? AND session=?",
                    (run_id, str(session)),
                )
            )
        metrics = recall_metrics(
            rankings,
            {r[0] for r in pool if r[1]},
            {r[0] for r in pool},
            hits,
            components,
            metadata["shadow_recipes"],
            datetime.fromisoformat(metadata["cutoff"]),
        )
        metrics["broad_references"] = len(rows)
        metrics["eligible_population"] = len(population)
        metrics["ineligible_references"] = sum(r["eligibility"] == "INELIGIBLE" for r in rows)
        if metadata["recipe"].get("transport_parity"):
            five = {
                json.loads(r["identity"])["con_id"]: decoded_bars(r["five_minute_bar"])
                for r in rows
                if r["identity"]
            }
            metrics["transport_parity"] = transport_parity(
                population, bars, five, prefix, MarketId(metadata["market"])
            )
        self.store.complete_oracle(run_id, session, rankings, metrics)
