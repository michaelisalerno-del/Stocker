"""Method catalogue: each package owns search, qualification, entry and exit policy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from stocker_core.candidate_selection import (
    SESSION_HARD_CANDIDATE_RECIPE,
    candidate_evidence_status,
)
from stocker_core.discovery import SESSION_HARD_DISCOVERY, DiscoveryProfile
from stocker_core.markets import (
    MARKET_CATALOGUE,
    CapBucket,
    MarketId,
    MarketUniverseSpec,
    get_market,
)
from stocker_core.universes import UniverseDefinition

ARTIFACTS = Path(__file__).parent / "method_artifacts" / "session_hard"
PROSPECTIVE_SPEC_SHA256 = "de2c4b3b90e9ecfff44cc7da971b2d9700eb277ece423bab6a4ed8e1fb3b6965"
SESSION_HARD_SCREEN_LIMIT = 50


def content_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verified_q1_spec() -> dict[str, Any]:
    raw = (ARTIFACTS / "prospective_q1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROSPECTIVE_SPEC_SHA256:
        raise ValueError("Session HARD prospective admission specification hash mismatch")
    spec: dict[str, Any] = json.loads(raw)
    for name, field in (
        ("MODEL_T0.joblib", "model_sha256"),
        ("MODEL_T0_parameters.json", "preprocessing_parameters_sha256"),
        ("fixed_spec.json", "research_spec_sha256"),
        ("fit_score_distribution.json", "score_distribution_sha256"),
    ):
        if hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest() != spec[field]:
            raise ValueError(f"Session HARD required artifact hash mismatch: {name}")
    return spec


def session_hard_universe(
    listings: Sequence[UniverseDefinition], selected: MarketId
) -> UniverseDefinition:
    market = get_market(selected)
    return UniverseDefinition(
        universe_id=f"{selected.value}_METHOD_ACTIVITY",
        name=market.display_name,
        market_spec=MarketUniverseSpec(market_id=selected, cap_bucket=CapBucket.ALL),
    )


@dataclass(frozen=True, slots=True)
class MethodDefinition:
    method_id: str
    version: str
    config_name: str
    label: str
    supported_markets: tuple[MarketId, ...]
    environments: tuple[str, ...]
    specification_builder: Callable[[MarketId], dict[str, Any]]
    universe_builder: Callable[[Sequence[UniverseDefinition], MarketId], UniverseDefinition]
    discovery_profiles: tuple[tuple[MarketId, DiscoveryProfile], ...] = ()

    def discovery_profile(self, market: MarketId) -> DiscoveryProfile | None:
        return dict(self.discovery_profiles).get(market)

    @property
    def strategy_id(self) -> str:
        return self.method_id

    @property
    def strategy_version(self) -> str:
        return self.version

    @property
    def display_token(self) -> str:
        return self.label

    def specification(self, market: MarketId | str) -> dict[str, Any]:
        selected = MarketId(market)
        if selected not in self.supported_markets:
            raise ValueError(f"{self.label} is not supported for {selected.value}")
        return self.specification_builder(selected)


def legacy_session_hard_specification(selected: MarketId) -> dict[str, Any]:
    q1 = verified_q1_spec()
    spec: dict[str, Any] = {
        "method_id": LEGACY_SESSION_HARD.method_id,
        "method_version": LEGACY_SESSION_HARD.version,
        "market": selected.value,
        "universe_search": {
            "builder": "IBKR_ACTIVITY_LIQUIDITY_V2",
            "activity_profile": "ACTIVITY_LIQUIDITY_V2",
            "components": ["TOP_TRADE_RATE", "MOST_ACTIVE_AVG_USD", "HOT_BY_VOLUME"],
            "stock_type_filter": "CORP",
            "verified_stock_types": ["COMMON", "CORP", "ADR", "REIT"],
            "capture_active_minutes": 15,
            "capture_policy": (
                "First available capture at/after minute 15; actual timestamp; no replay"
            ),
            "watch_limit": SESSION_HARD_SCREEN_LIMIT,
            "capacity_scope": "Per-market screening pool, independent of live trade-feed capacity",
            "ranking": "Scanner hit count, equal-weight rank sum, best rank, symbol, conId",
            "coverage": (
                "IBKR corporation-only liquidity/activity shortlist in every market; "
                "US authoritative membership enforced"
            ),
            "validation": "UNVALIDATED_ACTIVITY_FILTER_PAPER_TEST",
            "market": selected.value,
            "cap_constraint": None,
            "manual_universe": "research/testing input only",
            "screen": "IBKR_STOCK_QUALIFICATION_AND_REQUIRED_DATA",
            "suitability": "No additional validated stock suitability filter",
            "research_only_diagnostics": ["market_cap", "volatility", "liquidity"],
        },
        "data_requirements": {
            "source": "IBKR",
            "daily": "21 consecutive prior-session final RTH minute closes",
            "intraday": "Exact completed RTH 1-minute prefix; 5-minute aggregation; T0 open",
            "hv": "20 prior close-to-close log returns; sample std ddof=1 * sqrt(252)",
            "warmup": "Required history and prior-session cohort; frozen missing preprocessing",
            "prefetch_lead_minutes": 5,
            "events": "Ordered TRADES events; completed-minute extrema cannot choose direction",
        },
        "qualification": {
            "scorer": "Frozen Session HARD Model B",
            "score_min": 0.999361477,
            "pre_move_min_exclusive": 0.475764059845861,
            "checkpoints": list(range(6, 35, 2)),
        },
        "vetoes": {
            "cohort": "Original causal T0 NON_MID population",
            "whipsaw": "Frozen MODEL_T0",
            "q1_risk_cutoff": q1["q1_risk_cutoff"],
            "comparison": "<=",
            "q1_spec_sha256": PROSPECTIVE_SPEC_SHA256,
        },
        "ranking_capacity": (
            "50 stocks screened per market; independent shared account/feed capacity applies; "
            "missing causal trade streams prevent entry and remain visible"
        ),
        "direction": "First UP break LONG; first DOWN break SHORT; no reversal",
        "entry": {"trigger_M": 0.20, "window_minutes": 5, "reference": "Exact threshold"},
        "exits": {"stop_M": 0.50, "target_M": 1.00, "deadline_minutes_from_t0": 15},
        "economics": {
            "round_trip_research_cost_bps": 10,
            "admission": "No pooled legacy payoff hurdle in this frozen candidate",
        },
        "execution": {
            "shortability": "Required for SHORT",
            "orders": "Protected limit entry",
            "account_controls": "Existing exposure, permissions, broker/account readiness",
        },
        "session": "Market calendar regular trading hours; existing holidays and breaks",
        "runtime_state": "Run-scoped candidates, frozen inputs, first break and lifecycle",
        "artifact_hashes": {
            "model": q1["model_sha256"],
            "preprocessing": q1["preprocessing_parameters_sha256"],
            "research_spec": q1["research_spec_sha256"],
            "fit_distribution": q1["score_distribution_sha256"],
            "prospective_q1": PROSPECTIVE_SPEC_SHA256,
        },
    }
    profile = LEGACY_SESSION_HARD.discovery_profile(selected)
    if profile is not None:
        spec["universe_search"] = {
            "builder": "DYNAMIC_IBKR",
            "activity_profile": profile.profile_id,
            "discovery_profile": profile.model_dump(mode="json"),
            "capture_active_minutes": 15,
            "capture_policy": "Once per run/session; explicit rebuild while disabled",
            "capacity_scope": "Resource watch pool; independent of trade ranking/feed limits",
            "coverage": "Five canonical cap bands; broker identities; no saved listing filter",
            "cap_constraint": None,
            "validation": "UNVALIDATED_ACTIVITY_FILTER_PAPER_TEST",
        }
        spec["ranking_capacity"] = (
            "Configurable discovery watch pool; independent shared account/feed capacity applies; "
            "missing causal trade streams prevent entry and remain visible"
        )
    if get_market(selected).listing_membership is None:
        # Sharing a discovery profile is not evidence of model transfer.
        spec["universe_search"].update(
            {
                "validation": "UNVALIDATED_CROSS_MARKET_PAPER_TEST",
            }
        )
    return spec


LEGACY_SESSION_HARD = MethodDefinition(
    method_id="SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D",
    version="SESSION_HARD_CAUSAL_Q1_DISCOVERY_V7",
    config_name="SESSION_HARD",
    label="Session HARD",
    supported_markets=(
        MarketId.US_ALL,
        *(m.market_id for m in MARKET_CATALOGUE if m.market_id is not MarketId.US_ALL),
    ),
    environments=("PAPER",),
    specification_builder=legacy_session_hard_specification,
    universe_builder=session_hard_universe,
    discovery_profiles=tuple(
        (
            market.market_id,
            SESSION_HARD_DISCOVERY.model_copy(update={
                # Advertised Gateway locations; scope NASDAQ/NYSE/TSX before row limits.
                "scanner_location": {
                    MarketId.US_NASDAQ: "STK.NASDAQ",
                    MarketId.US_NYSE: "STK.NYSE",
                    MarketId.CANADA_TSX: "STK.NA.TSE",
                }.get(market.market_id, market.scanner_location),
            }),
        )
        for market in MARKET_CATALOGUE
    ),
)



def candidate_universe(
    listings: Sequence[UniverseDefinition], selected: MarketId
) -> UniverseDefinition:
    market = get_market(selected)
    source = next((u for u in listings if u.universe_id == market.listing_membership), None)
    if source is None and market.listing_membership is None:
        source = next(
            (
                u
                for u in listings
                if u.market_spec is not None
                and u.market_spec.market_id == selected
                and u.members
                and u.market_spec.cap_bucket is CapBucket.ALL
                and not u.universe_id.endswith("_METHOD_ACTIVITY")
            ),
            None,
        )
    return UniverseDefinition(
        universe_id=f"{selected.value}_OPENING_CANDIDATES_V1",
        name=market.display_name,
        members=source.members if source else (),
        market_spec=MarketUniverseSpec(market_id=selected, cap_bucket=CapBucket.ALL),
    )


def session_hard_specification(selected: MarketId) -> dict[str, Any]:
    # Trading specification is inherited byte-for-byte in value; only acquisition,
    # candidate reduction and operational version/capacity lineage change.
    spec = legacy_session_hard_specification(selected)
    market = get_market(selected)
    recipe = json.loads(json.dumps(asdict(SESSION_HARD_CANDIDATE_RECIPE)))
    recipe["stages"] = [
        dict(
            asdict(stage),
            offset_active_minutes=stage.minutes,
            formula=(
                "(max(high)-min(low))/first_open"
                if stage.feature == "RANGE"
                else "sqrt(log(C0/O0)^2 + sum(log(Cj/Cj-1)^2))"
            ),
        )
        for stage in SESSION_HARD_CANDIDATE_RECIPE.stages
    ]
    recipe.update(
        missing_policy="MISSING_LAST",
        tie_policy="ASCENDING_SHA256_SYMBOL",
        evidence_status=candidate_evidence_status(selected).value,
        cross_market_evidence=(
            "US_DEVELOPMENT_POPULATION"
            if market.country == "US"
            else "UNVALIDATED_CROSS_MARKET_PAPER_TRANSFER"
        ),
    )
    spec["method_version"] = SESSION_HARD.version
    spec["candidate_selection"] = recipe
    spec["universe_search"] = {
        "builder": "BROAD_ELIGIBLE_MARKET_UNIVERSE",
        "source": "AUTHORITATIVE_LISTINGS"
        if market.listing_membership
        else "CACHED_MARKET_UNIVERSE",
        "cap_constraint": None,
        "eligibility": ["COMMON", "CORP", "ADR", "REIT"],
        "scanner_source_evidence": "UNVALIDATED_UPSTREAM_ACQUISITION",
        "missing_source": "BROAD_UNIVERSE_UNAVAILABLE",
        "opening_data_failure": "BROAD_OPENING_DATA_CAPACITY_UNRESOLVED",
        "calendar": market.calendar,
        "timezone": market.timezone,
    }
    spec["ranking_capacity"] = "Frozen candidate stages; existing shared account/feed capacity"
    return spec


SESSION_HARD = replace(
    LEGACY_SESSION_HARD,
    version="SESSION_HARD_CAUSAL_Q1_CANDIDATES_V8",
    specification_builder=session_hard_specification,
    universe_builder=candidate_universe,
    discovery_profiles=(),
)


def runnable_methods() -> tuple[MethodDefinition, ...]:
    """Existing V7 runs keep their operational behavior; only V8 is selectable."""
    return (SESSION_HARD, LEGACY_SESSION_HARD)


def installed_methods() -> tuple[MethodDefinition, ...]:
    return (SESSION_HARD,)


def get_method(method_id: str, version: str) -> MethodDefinition:
    for method in runnable_methods():
        if (method.method_id, method.version) == (method_id, version):
            return method
    raise ValueError(f"Method is historical-only or unknown: {method_id}/{version}")


def validate_run_method(run: Any) -> None:
    """Admission boundary for starting/resuming; historical records remain loadable."""
    method = get_method(str(run.strategy_id), str(run.strategy_version))
    if run.market_id is None or run.method_spec is None:
        raise ValueError("Historical method runs cannot start; create a current method run")
    spec = method.specification(run.market_id)
    if (
        run.strategy != method.config_name
        or run.method_spec != spec
        or run.method_spec_hash != content_hash(spec)
        or run.environment.value not in method.environments
        or run.universe_snapshot is None
        or run.universe_snapshot.universe_id != run.universe
        or run.universe_snapshot.market_spec is None
        or run.universe_snapshot.market_spec.market_id != run.market_id
        or run.screen is not None
        or run.cap_bucket is not CapBucket.ALL
    ):
        raise ValueError("Run does not match its installed method package")
