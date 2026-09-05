"""Backend-owned market run construction for the Universes page."""

from __future__ import annotations

from hashlib import sha256

from stocker_core.config import RunsConfig
from stocker_core.markets import (
    CAP_BUCKETS_V1,
    MARKET_CATALOGUE,
    CapBucket,
    MarketId,
    MarketUniverseSpec,
    get_market,
)
from stocker_core.runs import (
    ACTIVITY_SHORTLIST_V1_ACTIVE_MINUTES,
    ACTIVITY_SHORTLIST_V1_ID,
    ACTIVITY_SHORTLIST_V1_VERSION,
    ACTIVITY_SHORTLIST_V1_WATCH_LIMIT,
    CandidateScreen,
    Environment,
    RunConfig,
    RunRiskConfig,
    RunScreenConfig,
    RunWindow,
)
from stocker_core.strategies import get_strategy, installed_strategies
from stocker_core.universes import UniverseDefinition


class UniverseRunBuilder:
    """Create, disable, or re-enable one immutable semantic run lineage."""

    def options(self) -> dict[str, object]:
        return {
            "markets": [
                {
                    "market_id": item.market_id.value,
                    "label": item.display_name,
                    "region": item.region,
                    "currency": item.currency,
                    "calendar": item.calendar,
                    "timezone": item.timezone,
                    "session": " / ".join(
                        f"{segment.opens_at.strftime('%H:%M')}–{segment.closes_at.strftime('%H:%M')}"
                        for segment in item.regular_sessions
                    ),
                    "scanner_readiness": "BROKER_NOT_CONNECTED",
                    "experimental": item.experimental,
                }
                for item in MARKET_CATALOGUE
            ],
            "capitalisation": [
                {
                    "cap_bucket": item.bucket.value,
                    "label": item.label,
                    "minimum_usd": item.minimum_usd,
                    "maximum_usd_exclusive": item.maximum_usd_exclusive,
                    "version": CAP_BUCKETS_V1.version,
                }
                for item in CAP_BUCKETS_V1.buckets
            ],
            "strategies": [
                {
                    "strategy_id": item.strategy_id,
                    "strategy_version": item.strategy_version,
                    "label": item.label,
                    "environments": list(item.environments),
                }
                for item in installed_strategies()
            ],
            "candidate_screen": {
                "screen_id": ACTIVITY_SHORTLIST_V1_ID,
                "screen_version": ACTIVITY_SHORTLIST_V1_VERSION,
                "label": "Activity Shortlist V1",
                "screen_active_minutes_after_open": ACTIVITY_SHORTLIST_V1_ACTIVE_MINUTES,
                "watch_limit": ACTIVITY_SHORTLIST_V1_WATCH_LIMIT,
            },
        }

    def add(
        self,
        config: RunsConfig,
        *,
        market_id: MarketId | str,
        cap_bucket: CapBucket | str,
        strategy_id: str,
        strategy_version: str,
        environment: Environment,
        risk: RunRiskConfig | None = None,
    ) -> tuple[RunsConfig, RunConfig]:
        market = get_market(market_id)
        cap = CapBucket(cap_bucket)
        method = get_strategy(strategy_id, strategy_version)
        if environment.value not in method.environments:
            raise ValueError(f"{method.strategy_version} is PAPER-only")
        identity = self._identity(
            market.market_id,
            cap,
            strategy_id,
            strategy_version,
            ACTIVITY_SHORTLIST_V1_ID,
            ACTIVITY_SHORTLIST_V1_VERSION,
            environment,
        )
        existing = next(
            (item for item in config.runs if self._run_identity(item) == identity), None
        )
        if existing is not None:
            if existing.enabled:
                return config, existing
            enabled = existing.model_copy(update={"enabled": True})
            return self._replace_run(config, enabled), enabled

        if environment is Environment.LIVE:
            paper_identity = (*identity[:-1], Environment.PAPER.value)
            if not any(self._run_identity(item) == paper_identity for item in config.runs):
                raise ValueError("LIVE requires an exact matching PAPER run")

        universe_id = f"{market.market_id.value}_{cap.value}_{CAP_BUCKETS_V1.version}"
        universe = next(
            (item for item in config.universes if item.universe_id == universe_id), None
        )
        if universe is None:
            listing_members = next(
                (
                    item.members
                    for item in config.universes
                    if market.listing_membership is not None
                    and item.universe_id == market.listing_membership
                ),
                (),
            )
            if market.market_id in {MarketId.US_NASDAQ, MarketId.US_NYSE} and not listing_members:
                raise ValueError(
                    f"{market.market_id.value} requires authoritative listing membership"
                )
            universe = UniverseDefinition(
                universe_id=universe_id,
                name=f"{market.display_name} {CAP_BUCKETS_V1.definition(cap).label}",
                members=listing_members,
                market_spec=MarketUniverseSpec(
                    market_id=market.market_id,
                    cap_bucket=cap,
                    cap_bucket_version=CAP_BUCKETS_V1.version,
                ),
            )
        display_name = f"{market.short_name} · {method.display_token} · {cap.value}"
        digest = sha256("|".join(identity).encode("utf-8")).hexdigest()[:12]
        run = RunConfig(
            run_id=f"{market.market_id.value.lower()}-{method.display_token.lower()}-{cap.value.lower()}-{environment.value.lower()}-{digest}",
            enabled=True,
            universe=universe_id,
            strategy=method.config_name,
            strategy_id=method.strategy_id,
            strategy_version=method.strategy_version,
            market_id=market.market_id,
            cap_bucket=cap,
            cap_bucket_version=CAP_BUCKETS_V1.version,
            candidate_screen_id=ACTIVITY_SHORTLIST_V1_ID,
            candidate_screen_version=ACTIVITY_SHORTLIST_V1_VERSION,
            display_name=display_name,
            environment=environment,
            risk=risk,
            session=RunWindow(
                start=market.regular_sessions[0].opens_at,
                end=market.regular_sessions[-1].closes_at,
                timezone=market.timezone,
                calendar=market.calendar,
            ),
            screen=RunScreenConfig(
                method=CandidateScreen.ACTIVITY_SHORTLIST_V1,
                max_results=ACTIVITY_SHORTLIST_V1_WATCH_LIMIT,
                version=ACTIVITY_SHORTLIST_V1_VERSION,
                scheduled_active_minutes=ACTIVITY_SHORTLIST_V1_ACTIVE_MINUTES,
            ),
        )
        universes = (
            config.universes if universe in config.universes else (*config.universes, universe)
        )
        return RunsConfig(
            session_hard_hv_round_trip_cost_bps=config.session_hard_hv_round_trip_cost_bps,
            universes=universes,
            runs=(*config.runs, run),
        ), run

    def disable(self, config: RunsConfig, run_id: str) -> RunsConfig:
        run = next((item for item in config.runs if item.run_id == run_id), None)
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        return self._replace_run(config, run.model_copy(update={"enabled": False}))

    @staticmethod
    def _replace_run(config: RunsConfig, selected: RunConfig) -> RunsConfig:
        return RunsConfig(
            session_hard_hv_round_trip_cost_bps=config.session_hard_hv_round_trip_cost_bps,
            universes=config.universes,
            runs=tuple(
                selected if item.run_id == selected.run_id else item for item in config.runs
            ),
        )

    @staticmethod
    def _identity(
        market_id: MarketId,
        cap_bucket: CapBucket,
        strategy_id: str,
        strategy_version: str,
        screen_id: str,
        screen_version: str,
        environment: Environment,
    ) -> tuple[str, ...]:
        return (
            market_id.value,
            cap_bucket.value,
            CAP_BUCKETS_V1.version,
            strategy_id,
            strategy_version,
            screen_id,
            screen_version,
            environment.value,
        )

    @staticmethod
    def _run_identity(run: RunConfig) -> tuple[str, ...] | None:
        if None in {
            run.market_id,
            run.cap_bucket,
            run.cap_bucket_version,
            run.strategy_id,
            run.strategy_version,
            run.candidate_screen_id,
            run.candidate_screen_version,
        }:
            return None
        assert run.market_id is not None
        assert run.cap_bucket is not None
        return (
            run.market_id.value,
            run.cap_bucket.value,
            str(run.cap_bucket_version),
            str(run.strategy_id),
            str(run.strategy_version),
            str(run.candidate_screen_id),
            str(run.candidate_screen_version),
            run.environment.value,
        )
