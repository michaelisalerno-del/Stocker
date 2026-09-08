"""The explicit composition seam between method packages and shared runtime services."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from stocker_core.markets import MarketId
from stocker_core.methods import SESSION_HARD, get_method
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


def session_hard_services(
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
    )


# Add a method's engine and data composition here; UI never branches on its name.
_PACKAGES = {
    (SESSION_HARD.method_id, SESSION_HARD.version): (SessionHardMethod, session_hard_services),
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
    return engine(market, clock=clock)


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
