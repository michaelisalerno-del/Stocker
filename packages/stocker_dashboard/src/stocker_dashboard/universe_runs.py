"""Market -> Method -> Run construction; legacy cap lineage remains readable."""

from __future__ import annotations

from stocker_core.config import RunsConfig
from stocker_core.discovery import UniverseSource
from stocker_core.markets import CapBucket, MarketId, get_market
from stocker_core.methods import content_hash, get_method, installed_methods
from stocker_core.runs import Environment, RunConfig, RunRiskConfig, RunWindow


class UniverseRunBuilder:
    def options(self) -> dict[str, object]:
        markets = dict.fromkeys(
            m for method in installed_methods() for m in method.supported_markets
        )
        return {
            "markets": [
                {
                    "market_id": m.value,
                    "label": get_market(m).display_name,
                    "region": get_market(m).region,
                    "currency": get_market(m).currency,
                    "calendar": get_market(m).calendar,
                    "timezone": get_market(m).timezone,
                    "session": " / ".join(
                        f"{s.opens_at:%H:%M}–{s.closes_at:%H:%M}"
                        for s in get_market(m).regular_sessions
                    ),
                    "scanner_readiness": "BROAD_UNIVERSE_REQUIRED",
                    "experimental": get_market(m).listing_membership is None,
                    "search_policy": "Range5 250 → RV10 50 → RV15 30 · broad universe required",
                    "validation": (
                        "US development candidate"
                        if get_market(m).listing_membership
                        else "UNVALIDATED_CROSS_MARKET_PAPER_TRANSFER"
                    ),
                }
                for m in markets
            ],
            "strategies": [
                {
                    "strategy_id": method.method_id,
                    "strategy_version": method.version,
                    "label": method.label,
                    "environments": list(method.environments),
                    "supported_markets": [m.value for m in method.supported_markets],
                }
                for method in installed_methods()
            ],
            "candidate_screen": {
                "label": "Range5 HIGH250 → RV10 HIGH50 → RV15 HIGH30",
                "screen_active_minutes_after_open": 15,
            },
        }

    def add(
        self,
        config: RunsConfig,
        *,
        market_id: MarketId | str,
        strategy_id: str,
        strategy_version: str,
        environment: Environment,
        risk: RunRiskConfig | None = None,
        historical_reproduction: bool = False,
    ) -> tuple[RunsConfig, RunConfig]:
        method = get_method(strategy_id, strategy_version)
        if not historical_reproduction and method not in installed_methods():
            raise ValueError("Historical method versions cannot create new runs")
        market = get_market(market_id)
        spec = method.specification(market.market_id)
        if environment.value not in method.environments:
            raise ValueError(f"{method.label} is PAPER-only")
        digest = content_hash(spec)
        run_id = (
            f"{market.market_id.value.lower()}-{method.config_name.lower()}-"
            f"{environment.value.lower()}-{digest[:12]}"
        )
        existing = next((r for r in config.runs if r.run_id == run_id), None)
        if existing is not None:
            run = existing.model_copy(update={"enabled": True})
            return self._replace_run(config, run), run
        universe = method.universe_builder(config.universes, market.market_id)
        run = RunConfig(
            run_id=run_id,
            universe=universe.universe_id,
            strategy=method.config_name,
            strategy_id=method.method_id,
            strategy_version=method.version,
            market_id=market.market_id,
            cap_bucket=CapBucket.ALL,
            cap_bucket_version="CAP_BUCKETS_V1",
            candidate_screen_id="METHOD_REQUIRED_DATA",
            candidate_screen_version=method.version,
            display_name=f"{market.short_name} · {method.label}",
            environment=environment,
            risk=risk,
            session=RunWindow(
                start=market.regular_sessions[0].opens_at,
                end=market.regular_sessions[-1].closes_at,
                timezone=market.timezone,
                calendar=market.calendar,
            ),
            method_spec=spec,
            method_spec_hash=digest,
            universe_snapshot=universe,
            universe_source=(
                UniverseSource.SCANNER_ASSISTED_UNIVERSE_ACQUISITION
                if "universe_acquisition" in spec else
                UniverseSource.DYNAMIC_IBKR
                if method.discovery_profile(market.market_id) is not None else
                UniverseSource.AUTHORITATIVE_LISTINGS if market.listing_membership else
                UniverseSource.CACHED_MARKET_UNIVERSE
            ),
            discovery_profile=method.discovery_profile(market.market_id),
        )
        universes = tuple(u for u in config.universes if u.universe_id != universe.universe_id)
        return config.model_copy(
            update={"universes": (*universes, universe), "runs": (*config.runs, run)}
        ), run

    def disable(self, config: RunsConfig, run_id: str) -> RunsConfig:
        run = next((r for r in config.runs if r.run_id == run_id), None)
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        return self._replace_run(config, run.model_copy(update={"enabled": False}))

    @staticmethod
    def _replace_run(config: RunsConfig, selected: RunConfig) -> RunsConfig:
        return config.model_copy(
            update={
                "runs": tuple(selected if r.run_id == selected.run_id else r for r in config.runs)
            }
        )
