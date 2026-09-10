"""Independent run configuration and minimal in-memory lifecycle."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stocker_core.markets import CAP_BUCKETS_V1, CapBucket, MarketId
from stocker_core.universes import Identifier, UniverseCatalog, UniverseDefinition

ACTIVITY_CAPACITY_V2_VERSION = "ACTIVITY_CAPACITY_V3_SCREEN50"


def activity_snapshot_version(profile_id: str) -> str:
    return ACTIVITY_CAPACITY_V2_VERSION if profile_id == "ACTIVITY_CAPACITY_V2" else profile_id


class Environment(StrEnum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class RunWindow(BaseModel):
    """Timezone-aware market window consumed by the Stage 8 scheduler."""

    model_config = ConfigDict(frozen=True)

    start: time
    end: time
    timezone: Identifier
    calendar: Identifier = "XNYS"


class RunRiskConfig(BaseModel):
    """Explicit Stage 7 risk inputs owned by one run."""

    model_config = ConfigDict(frozen=True)

    risk_per_trade: float
    max_concurrent_positions: int | None = None


class CandidateScreen(StrEnum):
    """Broker-independent candidate screen selected by one run."""

    HOT_BY_VOLUME = "HOT_BY_VOLUME"
    ACTIVITY_SHORTLIST_V1 = "ACTIVITY_SHORTLIST_V1"


ACTIVITY_SHORTLIST_V1_ID = "ACTIVITY_SHORTLIST_V1"
ACTIVITY_SHORTLIST_V1_VERSION = "ACTIVITY_SHORTLIST_V1"
ACTIVITY_SHORTLIST_V1_WATCH_LIMIT = 50
ACTIVITY_SHORTLIST_V1_ACTIVE_MINUTES = 15


class RunScreenConfig(BaseModel):
    """One bounded pre-qualification screen for a broad run universe."""

    model_config = ConfigDict(frozen=True)

    method: CandidateScreen
    max_results: int = Field(default=50, ge=1, le=50)
    version: Identifier | None = None
    scheduled_active_minutes: int = Field(default=15, ge=0)

    @model_validator(mode="after")
    def validate_frozen_profile(self) -> "RunScreenConfig":
        if self.method is CandidateScreen.ACTIVITY_SHORTLIST_V1 and (
            self.max_results != ACTIVITY_SHORTLIST_V1_WATCH_LIMIT
            or self.version != ACTIVITY_SHORTLIST_V1_VERSION
            or self.scheduled_active_minutes != ACTIVITY_SHORTLIST_V1_ACTIVE_MINUTES
        ):
            raise ValueError(
                "ACTIVITY_SHORTLIST_V1 requires version ACTIVITY_SHORTLIST_V1, "
                "max_results 50, and scheduled_active_minutes 15"
            )
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: Identifier
    enabled: bool = True
    archived: bool = False
    universe: Identifier
    strategy: Identifier
    strategy_id: Identifier | None = None
    strategy_version: Identifier | None = None
    market_id: MarketId | None = None
    cap_bucket: CapBucket | None = None
    cap_bucket_version: Identifier | None = None
    candidate_screen_id: Identifier | None = None
    candidate_screen_version: Identifier | None = None
    display_name: Identifier | None = None
    method_spec: dict[str, Any] | None = None
    method_spec_hash: str | None = None
    universe_snapshot: UniverseDefinition | None = None

    @property
    def activity_profile_id(self) -> str:
        return str(
            (self.method_spec or {})
            .get("universe_search", {})
            .get("activity_profile", ACTIVITY_SHORTLIST_V1_ID)
        )

    @property
    def activity_profile_version(self) -> str:
        return activity_snapshot_version(self.activity_profile_id)

    @property
    def uses_activity_shortlist(self) -> bool:
        """A method-owned discovery profile, or a historical explicit screen."""
        return (
            (self.method_spec or {}).get("universe_search", {}).get("activity_profile")
            in {ACTIVITY_SHORTLIST_V1_ID, "ACTIVITY_CAPACITY_V2"}
        ) or (
            self.screen is not None and self.screen.method is CandidateScreen.ACTIVITY_SHORTLIST_V1
        )

    environment: Environment
    risk: RunRiskConfig | None = None
    session: RunWindow | None = None
    screen: RunScreenConfig | None = None

    @model_validator(mode="after")
    def validate_lineage(self) -> "RunConfig":
        from stocker_core.strategies import SESSION_HARD_HV_METHOD

        if self.archived and self.enabled:
            raise ValueError("Archived runs cannot be enabled")
        if self.environment is Environment.LIVE and (
            self.strategy
            in {
                SESSION_HARD_HV_METHOD.config_name,
                SESSION_HARD_HV_METHOD.strategy_id,
            }
            or self.strategy_version == SESSION_HARD_HV_METHOD.strategy_version
        ):
            raise ValueError(f"{SESSION_HARD_HV_METHOD.strategy_version} is PAPER-only")
        lineage = (
            self.market_id,
            self.cap_bucket,
            self.cap_bucket_version,
            self.strategy_id,
            self.strategy_version,
            self.candidate_screen_id,
            self.candidate_screen_version,
        )
        if any(value is not None for value in lineage) and any(value is None for value in lineage):
            raise ValueError("Generated market runs require complete immutable lineage")
        if self.market_id is not None:
            from stocker_core.strategies import get_strategy

            if self.cap_bucket_version != CAP_BUCKETS_V1.version:
                raise ValueError("Generated market runs require CAP_BUCKETS_V1")
            if self.method_spec is not None and self.archived:
                from stocker_core.methods import content_hash

                if self.method_spec_hash != content_hash(self.method_spec):
                    raise ValueError("Archived method specification hash mismatch")
            if self.method_spec is not None and not self.archived:
                method = get_strategy(str(self.strategy_id), str(self.strategy_version))
                from stocker_core.methods import content_hash

                expected = method.specification(self.market_id)
                if self.method_spec != expected or self.method_spec_hash != content_hash(expected):
                    raise ValueError("Run method specification does not match frozen package")
                if self.strategy != method.config_name:
                    raise ValueError("Run method identity mismatch")
                if self.cap_bucket is not CapBucket.ALL or self.screen is not None:
                    raise ValueError(
                        "Universe/search belongs to the method; cap/screen override rejected"
                    )
                if self.environment.value not in method.environments:
                    raise ValueError(f"{method.label} is PAPER-only")
        if (
            self.screen is not None
            and self.screen.method is CandidateScreen.ACTIVITY_SHORTLIST_V1
            and (
                self.candidate_screen_id != ACTIVITY_SHORTLIST_V1_ID
                or self.candidate_screen_version != ACTIVITY_SHORTLIST_V1_VERSION
            )
        ):
            raise ValueError("Activity shortlist screen requires matching immutable lineage")
        if self.candidate_screen_id == ACTIVITY_SHORTLIST_V1_ID and (
            self.candidate_screen_version != ACTIVITY_SHORTLIST_V1_VERSION
            or self.screen is None
            or self.screen.method is not CandidateScreen.ACTIVITY_SHORTLIST_V1
        ):
            raise ValueError("Activity shortlist lineage must match its frozen screen profile")
        return self

    @property
    def execution_environment(self) -> Environment:
        """Explicit per-run execution destination; never inferred from broker state."""

        return self.environment

    @property
    def effective_strategy_id(self) -> str:
        return str(self.strategy_id or self.strategy)

    @property
    def effective_candidate_screen_id(self) -> str | None:
        if self.candidate_screen_id is not None:
            return str(self.candidate_screen_id)
        return self.screen.method.value if self.screen is not None else None


class RunState(StrEnum):
    CONFIGURED = "CONFIGURED"
    ACTIVE = "ACTIVE"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class RunInstance:
    """One run's immutable configuration, universe reference, and current state."""

    config: RunConfig
    universe: UniverseDefinition
    state: RunState = RunState.CONFIGURED


class RunManager:
    """Manage independent in-memory run states without starting broker work."""

    def __init__(self, universes: UniverseCatalog, runs: Iterable[RunConfig]) -> None:
        self._runs: dict[str, RunInstance] = {}
        for config in runs:
            if config.run_id in self._runs:
                raise ValueError(f"Duplicate run_id: {config.run_id}")
            self._runs[config.run_id] = RunInstance(
                config=config,
                universe=config.universe_snapshot or universes.get_universe(config.universe),
            )

    def start_run(self, run_id: str) -> RunInstance:
        """Mark one run active without invoking data, strategy, or execution work."""

        current = self.get_run(run_id)
        active = replace(current, state=RunState.ACTIVE)
        self._runs[run_id] = active
        return active

    def stop_run(self, run_id: str) -> RunInstance:
        """Mark one run stopped without affecting any other run."""

        current = self.get_run(run_id)
        stopped = replace(current, state=RunState.STOPPED)
        self._runs[run_id] = stopped
        return stopped

    def get_run(self, run_id: str) -> RunInstance:
        """Return one managed run or reject an unknown identifier clearly."""

        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise ValueError(f"Unknown run: {run_id}") from exc

    def list_runs(self) -> tuple[RunInstance, ...]:
        """List runs in deterministic configuration order."""

        return tuple(self._runs.values())
