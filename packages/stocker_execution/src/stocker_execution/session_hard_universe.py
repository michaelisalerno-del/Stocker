"""Method-owned watch identities feeding the existing contract/history boundary."""

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from stocker_core.markets import LIQUIDITY_ACTIVITY_COMPONENTS, CapBucket, get_market
from stocker_core.methods import SESSION_HARD_SCREEN_LIMIT
from stocker_core.runs import ACTIVITY_LIQUIDITY_V2_ID, RunInstance
from stocker_execution.activity_shortlist import (
    ActivityShortlistService,
    ActivityShortlistSnapshot,
    ActivityShortlistStatus,
    ActivityShortlistStore,
)
from stocker_execution.discovery import CandidateDiscovery, DiscoveryStore, watch_identities
from stocker_execution.ibkr import IbkrConnection, QualifiedInstrument
from stocker_execution.stage5 import (
    Stage5IneligibleInstrument,
    Stage5Membership,
    Stage5QualificationResult,
    qualify_active_runs,
)


class SessionHardUniverseSearch:
    def __init__(self, broker: IbkrConnection, database: Path, clock: Callable[[], datetime]):
        self.broker = broker
        self.clock = clock
        self.discovery = CandidateDiscovery(broker, DiscoveryStore(database), clock)
        self.activity = ActivityShortlistService(
            ActivityShortlistStore(database),
            profile_id=ACTIVITY_LIQUIDITY_V2_ID,
            profile_version=ACTIVITY_LIQUIDITY_V2_ID,
            watch_limit=SESSION_HARD_SCREEN_LIMIT,
            allow_late_capture=True,
            components=LIQUIDITY_ACTIVITY_COMPONENTS,
            stock_type_filter="CORP",
        )

    async def qualify(self, runs: Sequence[RunInstance]) -> Stage5QualificationResult:
        from stocker_execution.runtime import ExchangeSessionResolver

        now = self.clock()
        snapshots: dict[str, ActivityShortlistSnapshot] = {}
        identities: dict[str, tuple[QualifiedInstrument, ...]] = {}
        failures: list[Stage5IneligibleInstrument] = []
        for instance in runs:
            run = instance.config
            if not run.uses_activity_shortlist:
                continue
            assert run.market_id is not None
            market = ExchangeSessionResolver().resolve(run, now)
            definition = get_market(run.market_id)
            if run.uses_dynamic_discovery:
                if len(market.active_bar_starts) <= 3:
                    failures.append(
                        Stage5IneligibleInstrument(
                            run.activity_profile_id,
                            (Stage5Membership(run.run_id, instance.universe.universe_id),),
                            "SCANNER_NOT_AVAILABLE: no regular session",
                        )
                    )
                    identities[run.run_id] = ()
                    continue
                document = await self.discovery.discover(
                    run,
                    definition,
                    market.session,
                    now,
                    market.active_bar_starts[3],
                )
                identities[run.run_id] = watch_identities(document)
                if document["status"] != "READY":
                    failures.append(
                        Stage5IneligibleInstrument(
                            run.activity_profile_id,
                            (Stage5Membership(run.run_id, instance.universe.universe_id),),
                            "SCHEDULED"
                            if document["status"] == "SCHEDULED"
                            else f"{document['status']}: {document['reason']}",
                        )
                    )
                continue
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
        result = await qualify_active_runs(
            self.broker,
            runs,
            activity_snapshots=snapshots,
            candidate_identities=identities,
        )
        return Stage5QualificationResult(result.requests, (*result.ineligible, *failures))
