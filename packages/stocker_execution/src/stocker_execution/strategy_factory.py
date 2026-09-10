"""The explicit composition seam between method packages and shared runtime services."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from stocker_core.markets import MarketId
from stocker_core.methods import (
    LEGACY_SESSION_HARD,
    SESSION_HARD,
    SESSION_HARD_CANDIDATES_V8,
    get_method,
)
from stocker_core.runs import RunInstance
from stocker_execution.session_hard_method import SessionHardMethod

if TYPE_CHECKING:
    from stocker_execution.history import IbkrHistoryCache
    from stocker_execution.ibkr import IbkrConnection
    from stocker_execution.runtime import EntryBarSource, MarketSession, StrategyContextProvider
    from stocker_execution.stage5 import (
        Stage5Analyzer,
        Stage5QualificationResult,
        Stage5SnapshotStore,
    )


@dataclass(frozen=True)
class MethodServices:
    features: Stage5Analyzer
    context: StrategyContextProvider
    entries: EntryBarSource
    checkpoints: Callable[[MarketSession], tuple[tuple[int, datetime], ...]]
    qualify: Callable[[Sequence[RunInstance]], Awaitable[Stage5QualificationResult]] | None = None
    prepare_history_on_ready: bool = False
    incremental_checkpoints: bool = False
    universe_lifecycle: (
        Callable[
            [RunInstance, MarketSession, datetime], Awaitable[Stage5QualificationResult | None]
        ]
        | None
    ) = None
    universe_ready: Callable[[str, date], bool] | None = None
    universe_status: Callable[[str, date], dict[str, Any] | None] | None = None
    stop_universe: Callable[[], Awaitable[None]] | None = None
    background_work: Callable[[RunInstance, bool], Awaitable[None]] | None = None


def legacy_session_hard_services(
    broker: IbkrConnection,
    cache: IbkrHistoryCache,
    store: Stage5SnapshotStore,
    clock: Callable[[], datetime],
    logger: Any,
) -> MethodServices:
    from stocker_execution.session_hard_data import (
        IbkrSessionDataSource,
        PriorSessionExpectedMoveService,
    )
    from stocker_execution.session_hard_universe import SessionHardUniverseSearch
    from stocker_execution.stage5 import (
        STAGE5_HV_CALCULATION_VERSION,
        Stage5Analyzer,
        Stage5CurrentDataService,
    )

    source = IbkrSessionDataSource(broker, cache, logger=logger, clock=clock)
    current = Stage5CurrentDataService(
        broker,
        cache,
        PriorSessionExpectedMoveService(broker, cache),
        calculation_version=STAGE5_HV_CALCULATION_VERSION,
        clock=clock,
    )
    features = Stage5Analyzer(
        current,
        snapshot_store=store,
        calculation_version=STAGE5_HV_CALCULATION_VERSION,
    )
    checkpoints = tuple(SESSION_HARD.specification(MarketId.US_ALL)["qualification"]["checkpoints"])
    return MethodServices(
        features,
        source,
        source,
        lambda market: market.checkpoint_times(checkpoints),
        SessionHardUniverseSearch(broker, store.path, clock).qualify,
        prepare_history_on_ready=True,
        incremental_checkpoints=True,
    )


def session_hard_services(
    broker: IbkrConnection,
    cache: IbkrHistoryCache,
    store: Stage5SnapshotStore,
    clock: Callable[[], datetime],
    logger: Any,
) -> MethodServices:
    from dataclasses import replace

    from stocker_execution.candidate_pipeline import (
        CandidatePipeline,
        CandidateStore,
        ConfiguredUniverseProvider,
        OpeningBarSource,
    )

    services = legacy_session_hard_services(broker, cache, store, clock, logger)
    candidates = CandidatePipeline(
        CandidateStore(store.path),
        ConfiguredUniverseProvider(broker),
        OpeningBarSource(broker, cache),
        clock,
    )
    return replace(
        services,
        qualify=candidates.qualify,
        universe_lifecycle=candidates.advance,
        universe_ready=candidates.ready,
        universe_status=candidates.store.summary,
        stop_universe=candidates.stop,
    )


def acquired_session_hard_services(
    broker: IbkrConnection, cache: IbkrHistoryCache, store: Stage5SnapshotStore,
    clock: Callable[[], datetime], logger: Any,
) -> MethodServices:
    from dataclasses import replace

    from stocker_execution.acquired_candidates import AcquiredCandidates
    from stocker_execution.candidate_pipeline import CandidateStore

    services = legacy_session_hard_services(broker, cache, store, clock, logger)
    candidates = AcquiredCandidates(broker, cache, CandidateStore(store.path), clock)
    return replace(services, qualify=candidates.qualify, universe_lifecycle=candidates.advance,
                   universe_ready=candidates.ready, universe_status=candidates.summary,
                   stop_universe=candidates.stop, background_work=candidates.background)


# Add a method's engine and data composition here; UI never branches on its name.
_PACKAGES = {
    (SESSION_HARD.method_id, SESSION_HARD.version): (
        SessionHardMethod, acquired_session_hard_services,
    ),
    (SESSION_HARD_CANDIDATES_V8.method_id, SESSION_HARD_CANDIDATES_V8.version): (
        SessionHardMethod, session_hard_services,
    ),
    (LEGACY_SESSION_HARD.method_id, LEGACY_SESSION_HARD.version): (
        SessionHardMethod,
        legacy_session_hard_services,
    ),
}


def create_strategy(
    strategy_id: str,
    strategy_version: str,
    market: MarketId = MarketId.US_ALL,
    *,
    clock: Callable[[], datetime] | None = None,
) -> SessionHardMethod:
    try:
        engine, _ = _PACKAGES[(strategy_id, strategy_version)]
    except KeyError as exc:
        raise ValueError(f"Unsupported runtime strategy: {strategy_id}/{strategy_version}") from exc
    get_method(strategy_id, strategy_version).specification(market)
    return engine(market, clock=clock, method_version=strategy_version)


def create_method_services(
    strategy_id: str,
    strategy_version: str,
    broker: IbkrConnection,
    cache: IbkrHistoryCache,
    store: Stage5SnapshotStore,
    clock: Callable[[], datetime],
    logger: Any,
) -> MethodServices:
    _, build = _PACKAGES[(strategy_id, strategy_version)]
    return build(broker, cache, store, clock, logger)
