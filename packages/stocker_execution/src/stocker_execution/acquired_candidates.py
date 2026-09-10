"""Compose acquisition and audit around the unchanged candidate pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from typing import Any

from stocker_core.candidate_selection import CandidateIdentity
from stocker_core.runs import RunInstance
from stocker_execution.acquisition_store import AcquisitionStore
from stocker_execution.candidate_oracle import OracleAudit
from stocker_execution.candidate_pipeline import (
    STAGES,
    CandidatePipeline,
    CandidateStore,
    OpeningBarSource,
)
from stocker_execution.history import IbkrHistoryCache
from stocker_execution.ibkr import HistoricalBar, IbkrConnection, IbkrError
from stocker_execution.runtime import MarketSession
from stocker_execution.scanner_acquisition import ScannerAcquisition
from stocker_execution.stage5 import Stage5QualificationResult


class RecordedOpeningSource(OpeningBarSource):
    def __init__(
        self,
        shared: OpeningBarSource,
        store: AcquisitionStore,
        run_id: str,
        clock: Callable[[], datetime],
        phase: str = "LIVE",
    ):
        self.shared, self.store, self.run_id, self.clock, self.phase = (
            shared,
            store,
            run_id,
            clock,
            phase,
        )
        self.history = shared.history

    async def prefix(
        self,
        identity: CandidateIdentity,
        expected: tuple[datetime, ...],
        due: datetime,
        **kwargs: Any,
    ) -> tuple[HistoricalBar, ...]:
        from zoneinfo import ZoneInfo

        from stocker_core.markets import get_market

        stage = next(i for i, s in enumerate(STAGES) if s.minutes == len(expected))
        session = expected[0].astimezone(ZoneInfo(get_market(identity.market).timezone)).date()
        payload: dict[str, Any] = {
            "required_bars": len(expected),
            "boundary": due.isoformat(),
            "prefix_ready": False,
            "error": "",
        }
        try:
            bars = await self.shared.prefix(
                identity, expected, due, diagnostic=payload, use_cache=True
            )
            payload["prefix_ready"] = tuple(b.timestamp for b in bars) == expected
            payload["missing_prefix"] = not payload["prefix_ready"]
            payload["availability_delay_ms"] = (self.clock() - due).total_seconds() * 1000
            if stage == 0:
                payload["completed_before_next_stage"] = self.clock() < due + timedelta(
                    minutes=STAGES[1].minutes - STAGES[0].minutes
                )
            if self.phase == "LIVE" and not payload["prefix_ready"]:
                raise IbkrError("MISSING_REQUIRED_OPENING_PREFIX")
            return bars
        except BaseException as exc:
            payload["error"] = str(exc) or type(exc).__name__
            raise
        finally:
            await asyncio.to_thread(
                self.store.history_request,
                self.run_id,
                session,
                self.phase,
                stage,
                identity.con_id,
                payload,
            )


class AcquiredCandidates:
    def __init__(
        self,
        broker: IbkrConnection,
        cache: IbkrHistoryCache,
        candidate_store: CandidateStore,
        clock: Callable[[], datetime],
    ):
        self.store = AcquisitionStore(candidate_store.path)
        self.candidate_store, self.clock = candidate_store, clock
        self.provider = ScannerAcquisition(broker, self.store, clock)
        self.shared_source = OpeningBarSource(broker, cache)
        self.pipelines: dict[str, CandidatePipeline] = {}
        self.oracles: dict[str, OracleAudit] = {}
        self.broker = broker

    def pipeline(self, instance: RunInstance) -> CandidatePipeline:
        run_id = instance.config.run_id
        if run_id not in self.pipelines:
            source = RecordedOpeningSource(self.shared_source, self.store, run_id, self.clock)
            self.pipelines[run_id] = CandidatePipeline(
                self.candidate_store, self.provider, source, self.clock
            )
            self.oracles[run_id] = OracleAudit(
                self.broker,
                self.store,
                RecordedOpeningSource(self.shared_source, self.store, run_id, self.clock, "AUDIT"),
                self.clock,
            )
        return self.pipelines[run_id]

    async def qualify(self, runs: Sequence[RunInstance]) -> Stage5QualificationResult:
        results = await asyncio.gather(*(self.pipeline(i).qualify((i,)) for i in runs))
        return Stage5QualificationResult(
            tuple(r for result in results for r in result.requests),
            tuple(r for result in results for r in result.ineligible),
        )

    async def advance(
        self, instance: RunInstance, market: MarketSession, now: datetime
    ) -> Stage5QualificationResult | None:
        return await self.pipeline(instance).advance(instance, market, now)

    def ready(self, run_id: str, session: date) -> bool:
        row = self.candidate_store.summary(run_id, session)
        return bool(row and row["ready"])

    def summary(self, run_id: str, session: date) -> dict[str, Any] | None:
        return self.candidate_store.summary(run_id, session)

    async def background(self, instance: RunInstance, allowed: bool) -> None:
        self.pipeline(instance)
        await self.oracles[instance.config.run_id].tick(instance, allowed)

    async def stop(self) -> None:
        await asyncio.gather(*(p.stop() for p in self.pipelines.values()))
        await asyncio.gather(*(o.stop() for o in self.oracles.values()))
