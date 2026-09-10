"""Bound contract/history work using the same audited activity filter in every market."""

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from stocker_core.markets import CapBucket, get_market
from stocker_core.methods import SESSION_HARD_SCREEN_LIMIT
from stocker_core.runs import ACTIVITY_CAPACITY_V2_VERSION, RunInstance
from stocker_execution.activity_shortlist import (
    ActivityShortlistService,
    ActivityShortlistSnapshot,
    ActivityShortlistStatus,
    ActivityShortlistStore,
)
from stocker_execution.ibkr import IbkrConnection
from stocker_execution.stage5 import Stage5QualificationResult, qualify_active_runs


class SessionHardUniverseSearch:
    def __init__(self, broker: IbkrConnection, database: Path, clock: Callable[[], datetime]):
        self.broker = broker
        self.clock = clock
        self.activity = ActivityShortlistService(
            ActivityShortlistStore(database),
            profile_id="ACTIVITY_CAPACITY_V2",
            profile_version=ACTIVITY_CAPACITY_V2_VERSION,
            watch_limit=SESSION_HARD_SCREEN_LIMIT,
            allow_late_capture=True,
        )

    async def qualify(self, runs: Sequence[RunInstance]) -> Stage5QualificationResult:
        from stocker_execution.runtime import ExchangeSessionResolver

        now = self.clock()
        snapshots: dict[str, ActivityShortlistSnapshot] = {}
        for instance in runs:
            run = instance.config
            if not run.uses_activity_shortlist:
                continue
            assert run.market_id is not None
            market = ExchangeSessionResolver().resolve(run, now)
            definition = get_market(run.market_id)
            if len(market.active_bar_starts) <= 3:
                snapshots[run.run_id] = ActivityShortlistSnapshot(
                    market_id=run.market_id.value,
                    cap_bucket=CapBucket.ALL,
                    cap_bucket_version="CAP_BUCKETS_V1",
                    session=market.session,
                    screen_timestamp=now,
                    profile_id=self.activity.profile_id,
                    profile_version=self.activity.profile_version,
                    status=ActivityShortlistStatus.SCANNER_NOT_AVAILABLE,
                    components=(),
                    candidates=(),
                    reason="SCANNER_NOT_AVAILABLE",
                )
                continue
            snapshots[run.run_id] = await self.activity.get_or_create(
                self.broker,
                market=definition,
                cap_bucket=CapBucket.ALL,
                session=market.session,
                screen_at=market.active_bar_starts[3],
                now=now,
                allowed_symbols=(
                    frozenset(member.symbol for member in instance.universe.members)
                    if definition.listing_membership is not None
                    else None
                ),
            )
        return await qualify_active_runs(self.broker, runs, activity_snapshots=snapshots)
