"""Independent run configuration and minimal in-memory lifecycle."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from stocker_core.universes import Identifier, UniverseCatalog, UniverseDefinition


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


class RunScreenConfig(BaseModel):
    """One bounded pre-qualification screen for a broad run universe."""

    model_config = ConfigDict(frozen=True)

    method: CandidateScreen
    max_results: int = Field(default=50, ge=1, le=50)


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: Identifier
    enabled: bool = True
    universe: Identifier
    strategy: Identifier
    environment: Environment
    risk: RunRiskConfig | None = None
    session: RunWindow | None = None
    screen: RunScreenConfig | None = None

    @property
    def execution_environment(self) -> Environment:
        """Explicit per-run execution destination; never inferred from broker state."""

        return self.environment


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
                universe=universes.get_universe(config.universe),
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
