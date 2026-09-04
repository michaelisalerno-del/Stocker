"""Small Stage 8 application runtime over the completed Stage 1--7 seams."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

from stocker_core.config import IbkrConfig, RunsConfig, load_ibkr_config, load_runs_config
from stocker_core.logging import configure_logging
from stocker_core.markets import MARKET_CATALOGUE, ActivityScanner, CapBucket, get_market
from stocker_core.runs import CandidateScreen, Environment, RunConfig, RunInstance, RunManager
from stocker_core.strategies import SESSION_HARD_HV_METHOD, SESSION_HARD_METHOD
from stocker_core.universes import UniverseCatalog
from stocker_data.calendars import get_market_calendar
from stocker_execution.activity_shortlist import (
    ActivityShortlistService,
    ActivityShortlistSnapshot,
    ActivityShortlistStatus,
    ActivityShortlistStore,
)
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import BrokerAccountState, BrokerFill, OrderLifecycle
from stocker_execution.expected_move import (
    IbkrHistoricalVolatilityExpectedMoveService,
    PriorSessionContextExpectedMoveService,
)
from stocker_execution.history import (
    HistorySemantics,
    HistoryStatus,
    IbkrHistoryCache,
    IbkrHistoryService,
)
from stocker_execution.ibkr import (
    BrokerSession,
    CurrentQuote,
    IbkrConnection,
    IbkrError,
    IbkrResourceStatus,
    QualifiedInstrument,
    mask_ibkr_account,
)
from stocker_execution.pre_context import PriorSessionContextService, PriorSessionContextStore
from stocker_execution.session_hard_structure_d import (
    SESSION_HARD_CHECKPOINTS,
    CohortOpportunity,
    EntryBar,
    PreMoveBand,
    SessionHardAssessment,
    SessionHardStructureDStrategy,
    SignalStatus,
    StrategyContext,
    StrategyOpportunityKey,
    StrategySignal,
)
from stocker_execution.stage5 import (
    STAGE5_HV_CALCULATION_VERSION,
    Stage5Analyzer,
    Stage5CurrentDataService,
    Stage5FeatureSnapshot,
    Stage5IneligibleInstrument,
    Stage5Membership,
    Stage5QualificationResult,
    Stage5QualifiedRequest,
    Stage5SnapshotStore,
    Stage5Status,
    calculate_session_hard_inputs,
    qualify_active_runs,
)
from stocker_execution.stage7 import (
    ExecutionBroker,
    ExecutionDestination,
    ExecutionEnvironmentUnavailableError,
    ExecutionResultCode,
    ExecutionRouter,
    ReconciliationResult,
    Stage7ExecutionService,
    Stage7StrategyRuntime,
)
from stocker_execution.strategy_factory import create_strategy


class ApplicationState(StrEnum):
    STARTING = "STARTING"
    RECONCILING = "RECONCILING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class RunRuntimeState(StrEnum):
    DISABLED = "DISABLED"
    STARTING = "STARTING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"


class MarketSessionState(StrEnum):
    BEFORE_SESSION = "BEFORE_SESSION"
    ACTIVE_SESSION = "ACTIVE_SESSION"
    AFTER_SESSION = "AFTER_SESSION"
    CLOSED_DAY = "CLOSED_DAY"


class CheckpointState(StrEnum):
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    SKIPPED_MISSED = "SKIPPED_MISSED"
    SKIPPED_INTERRUPTED = "SKIPPED_INTERRUPTED"
    FAILED = "FAILED"


HV_EXPECTED_MOVE_PREFETCH_LEAD = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class MarketSession:
    session: date
    state: MarketSessionState
    opens_at: datetime | None
    closes_at: datetime | None
    active_bar_starts: tuple[datetime, ...] = ()

    def checkpoint_times(self) -> tuple[tuple[int, datetime], ...]:
        if self.opens_at is None or self.closes_at is None:
            return ()
        if self.active_bar_starts:
            return tuple(
                (checkpoint, self.active_bar_starts[checkpoint])
                for checkpoint in SESSION_HARD_CHECKPOINTS
                if checkpoint < len(self.active_bar_starts)
            )
        return tuple(
            (checkpoint, self.opens_at + timedelta(minutes=checkpoint * 5))
            for checkpoint in SESSION_HARD_CHECKPOINTS
            if self.opens_at + timedelta(minutes=checkpoint * 5) < self.closes_at
        )


@dataclass(frozen=True, slots=True)
class RuntimeCounters:
    checkpoints_processed: int = 0
    instruments_ready: int = 0
    signals: int = 0
    risk_rejects: int = 0
    orders: int = 0
    fills: int = 0
    broker_rejects: int = 0
    reconciliation_issues: int = 0
    disconnects: int = 0


@dataclass(frozen=True, slots=True)
class RunStatus:
    run_id: str
    universe: str
    strategy: str
    environment: Environment
    state: RunRuntimeState
    reason: str
    session: date | None
    market: MarketSessionState | None
    instruments_ready: int
    signals_today: int
    open_positions: int


@dataclass(frozen=True, slots=True)
class ExecutionEnvironmentStatus:
    environment: Environment
    connected: bool
    account: str | None
    expected_account: str
    reconciled: bool
    ready: bool
    equity: float | None = None
    buying_power: float | None = None


@dataclass(frozen=True, slots=True)
class ExecutionReadinessDiagnostic:
    environment: Environment
    connected: bool
    account: str | None
    expected_account: str
    expected_account_match: bool
    account_state_available: bool
    open_orders: int | None
    positions: int | None
    reconciled: bool
    ready: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    application: ApplicationState
    execution_environments: tuple[ExecutionEnvironmentStatus, ...]
    runs: tuple[RunStatus, ...]
    counters: RuntimeCounters
    ibkr_resources: IbkrResourceStatus | None = None

    @property
    def broker_connected(self) -> bool:
        """Stage 8 compatibility view of the PAPER broker."""

        selected = next(
            (item for item in self.execution_environments if item.environment is Environment.PAPER),
            None,
        )
        return selected.connected if selected is not None else False

    @property
    def account(self) -> str | None:
        """Stage 8 compatibility view of the PAPER account."""

        selected = next(
            (item for item in self.execution_environments if item.environment is Environment.PAPER),
            None,
        )
        return selected.account if selected is not None else None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "application": self.application.value,
            "ibkr": {
                item.environment.value: {
                    "connected": item.connected,
                    "account": mask_ibkr_account(item.account),
                    "expected_account": mask_ibkr_account(item.expected_account),
                    "reconciled": item.reconciled,
                    "ready": item.ready,
                    "equity": item.equity,
                    "buying_power": item.buying_power,
                }
                for item in self.execution_environments
            },
            "runs": [
                {
                    "run_id": run.run_id,
                    "universe": run.universe,
                    "strategy": run.strategy,
                    "environment": run.environment.value,
                    "state": run.state.value,
                    "reason": run.reason,
                    "session": run.session.isoformat() if run.session else None,
                    "market": run.market.value if run.market else None,
                    "instruments_ready": run.instruments_ready,
                    "signals_today": run.signals_today,
                    "open_positions": run.open_positions,
                }
                for run in self.runs
            ],
            "counters": {
                field: getattr(self.counters, field)
                for field in RuntimeCounters.__dataclass_fields__
            },
        }
        if self.ibkr_resources is not None:
            result["ibkr_resources"] = asdict(self.ibkr_resources)
        paper = next(
            (item for item in self.execution_environments if item.environment is Environment.PAPER),
            None,
        )
        if paper is not None:
            result["ibkr_paper"] = "connected" if paper.connected else "disconnected"
            result["account"] = mask_ibkr_account(paper.account)
        return result

    def as_text(self) -> str:
        lines = [f"Application: {self.application.value}"]
        for item in self.execution_environments:
            lines.extend(
                (
                    f"IBKR {item.environment.value}: "
                    f"{'connected' if item.connected else 'disconnected'}",
                    f"  account: {mask_ibkr_account(item.account) or 'unavailable'}",
                    f"  reconciled: {'yes' if item.reconciled else 'no'}",
                    f"  readiness: {'READY' if item.ready else 'NOT_READY'}",
                )
            )
        lines.append(
            f"Runs: {sum(run.state is RunRuntimeState.ACTIVE for run in self.runs)} active"
        )
        for run in self.runs:
            lines.extend(
                (
                    "",
                    f"{run.run_id}: {run.state.value}",
                    f"  universe: {run.universe}",
                    f"  strategy: {run.strategy}",
                    f"  environment: {run.environment.value}",
                    f"  instruments ready: {run.instruments_ready}",
                    f"  signals today: {run.signals_today}",
                    f"  open positions: {run.open_positions}",
                )
            )
            if run.reason:
                lines.append(f"  reason: {run.reason}")
        return "\n".join(lines)


class RuntimeBroker(ExecutionBroker, Protocol):
    async def connect(self) -> BrokerSession: ...

    def disconnect(self) -> None: ...

    def reconfigure(self, config: IbkrConfig) -> None: ...


class SessionResolver(Protocol):
    def resolve(self, run: RunConfig, now: datetime) -> MarketSession: ...


class StrategyContextProvider(Protocol):
    async def context_for(
        self,
        run: RunConfig,
        rows: Sequence[Stage5FeatureSnapshot],
        checkpoint: int,
        instruments: Mapping[int, QualifiedInstrument],
        cohort_history: Sequence[CohortOpportunity],
    ) -> StrategyContext: ...


class EntryBarSource(Protocol):
    async def bars_for(
        self,
        run: RunConfig,
        instruments: Mapping[int, QualifiedInstrument],
        *,
        session: date,
        now: datetime,
        signals: Sequence[StrategySignal],
    ) -> Mapping[int, Sequence[EntryBar]]: ...


Qualifier = Callable[[Sequence[RunInstance]], Awaitable[Stage5QualificationResult]]


class ExchangeSessionResolver:
    """Resolve a run's configured exchange day without using machine-local time."""

    def resolve(self, run: RunConfig, now: datetime) -> MarketSession:
        aware_now = _aware(now)
        if run.session is None:
            raise ValueError(f"run {run.run_id} requires an explicit market session")
        window = run.session
        timezone = ZoneInfo(str(window.timezone))
        local_now = aware_now.astimezone(timezone)
        session = local_now.date()
        calendar = get_market_calendar(str(window.calendar))
        schedule = calendar.schedule(start_date=session, end_date=session)
        if schedule.empty:
            return MarketSession(session, MarketSessionState.CLOSED_DAY, None, None)
        configured_open = datetime.combine(session, window.start, tzinfo=timezone).astimezone(UTC)
        configured_close = datetime.combine(session, window.end, tzinfo=timezone).astimezone(UTC)
        exchange_open = schedule.iloc[0]["market_open"].to_pydatetime().astimezone(UTC)
        exchange_close = schedule.iloc[0]["market_close"].to_pydatetime().astimezone(UTC)
        opens_at = max(configured_open, exchange_open)
        closes_at = min(configured_close, exchange_close)
        if closes_at <= opens_at:
            raise ValueError(f"run {run.run_id} session end must be after its start")
        if aware_now < opens_at:
            state = MarketSessionState.BEFORE_SESSION
        elif aware_now < closes_at:
            state = MarketSessionState.ACTIVE_SESSION
        else:
            state = MarketSessionState.AFTER_SESSION
        break_start = _optional_schedule_time(schedule.iloc[0].get("break_start"))
        break_end = _optional_schedule_time(schedule.iloc[0].get("break_end"))
        segments: tuple[tuple[datetime, datetime], ...] = ((opens_at, closes_at),)
        if (
            break_start is not None
            and break_end is not None
            and opens_at < break_start < break_end < closes_at
        ):
            segments = ((opens_at, break_start), (break_end, closes_at))
        active_bar_starts = tuple(
            timestamp
            for segment_open, segment_close in segments
            for timestamp in _five_minute_slots(segment_open, segment_close)
        )
        return MarketSession(session, state, opens_at, closes_at, active_bar_starts)


class RuntimeStore:
    """Durable Stage 8 checkpoints, reconciliation state, and burn-in counters."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runtime_checkpoints (
                    run_id TEXT NOT NULL,
                    session TEXT NOT NULL,
                    t0_utc TEXT NOT NULL,
                    checkpoint INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, session, t0_utc)
                );
                CREATE TABLE IF NOT EXISTS runtime_counters (
                    run_id TEXT NOT NULL,
                    session TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value INTEGER NOT NULL,
                    PRIMARY KEY (run_id, session, name)
                );
                CREATE TABLE IF NOT EXISTS runtime_reconciliation (
                    environment TEXT NOT NULL,
                    account TEXT NOT NULL,
                    connection_epoch INTEGER NOT NULL,
                    ok INTEGER NOT NULL,
                    detail TEXT NOT NULL,
                    reconciled_at TEXT NOT NULL,
                    PRIMARY KEY (environment, account)
                );
                CREATE TABLE IF NOT EXISTS runtime_cohort_opportunities (
                    identity TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    session TEXT NOT NULL,
                    pre_move_m REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runtime_signals (
                    signal_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    session TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def recover_interrupted(self, now: datetime) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runtime_checkpoints
                SET state = ?, detail = ?, updated_at = ?
                WHERE state = ?
                """,
                (
                    CheckpointState.SKIPPED_INTERRUPTED.value,
                    "previous process stopped during checkpoint; live opportunity not replayed",
                    _aware(now).isoformat(),
                    CheckpointState.PROCESSING.value,
                ),
            )
        return int(cursor.rowcount)

    def reserve_checkpoint(
        self, run_id: str, session: date, t0: datetime, checkpoint: int, now: datetime
    ) -> bool:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO runtime_checkpoints
                    (run_id, session, t0_utc, checkpoint, state, detail, updated_at)
                    VALUES (?, ?, ?, ?, ?, '', ?)
                    """,
                    (
                        run_id,
                        session.isoformat(),
                        _aware(t0).isoformat(),
                        checkpoint,
                        CheckpointState.PROCESSING.value,
                        _aware(now).isoformat(),
                    ),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def mark_checkpoint(
        self,
        run_id: str,
        session: date,
        t0: datetime,
        state: CheckpointState,
        detail: str,
        now: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runtime_checkpoints SET state = ?, detail = ?, updated_at = ?
                WHERE run_id = ? AND session = ? AND t0_utc = ?
                """,
                (
                    state.value,
                    detail,
                    _aware(now).isoformat(),
                    run_id,
                    session.isoformat(),
                    _aware(t0).isoformat(),
                ),
            )

    def mark_missed(
        self, run_id: str, session: date, t0: datetime, checkpoint: int, now: datetime
    ) -> bool:
        if not self.reserve_checkpoint(run_id, session, t0, checkpoint, now):
            return False
        self.mark_checkpoint(
            run_id,
            session,
            t0,
            CheckpointState.SKIPPED_MISSED,
            "runtime was not READY at T0; missed opportunity was not replayed",
            now,
        )
        return True

    def checkpoint_state(self, run_id: str, session: date, t0: datetime) -> CheckpointState | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT state FROM runtime_checkpoints
                WHERE run_id = ? AND session = ? AND t0_utc = ?
                """,
                (run_id, session.isoformat(), _aware(t0).isoformat()),
            ).fetchone()
        return CheckpointState(str(row["state"])) if row else None

    def increment(self, run_id: str, session: date, name: str, amount: int = 1) -> None:
        if name not in RuntimeCounters.__dataclass_fields__:
            raise ValueError(f"unknown runtime counter: {name}")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runtime_counters (run_id, session, name, value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (run_id, session, name)
                DO UPDATE SET value = value + excluded.value
                """,
                (run_id, session.isoformat(), name, amount),
            )

    def counters(self, run_id: str | None = None, session: date | None = None) -> RuntimeCounters:
        clauses: list[str] = []
        values: list[object] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if session is not None:
            clauses.append("session = ?")
            values.append(session.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT name, SUM(value) AS value FROM runtime_counters {where} GROUP BY name",
                values,
            ).fetchall()
        observed = {str(row["name"]): int(row["value"]) for row in rows}
        return RuntimeCounters(**observed)

    def record_reconciliation(
        self,
        *,
        environment: Environment,
        account: str,
        connection_epoch: int,
        ok: bool,
        detail: str,
        now: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runtime_reconciliation
                (environment, account, connection_epoch, ok, detail, reconciled_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (environment, account) DO UPDATE SET
                    connection_epoch = excluded.connection_epoch,
                    ok = excluded.ok,
                    detail = excluded.detail,
                    reconciled_at = excluded.reconciled_at
                """,
                (
                    environment.value,
                    account,
                    connection_epoch,
                    int(ok),
                    detail,
                    _aware(now).isoformat(),
                ),
            )

    def save_cohort(self, identity: str, opportunity: CohortOpportunity) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO runtime_cohort_opportunities
                (identity, run_id, session, pre_move_m) VALUES (?, ?, ?, ?)
                """,
                (
                    identity,
                    opportunity.run_id,
                    opportunity.session.isoformat(),
                    opportunity.pre_move_m,
                ),
            )

    def cohort_history(self, run_id: str) -> tuple[CohortOpportunity, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, session, pre_move_m
                FROM runtime_cohort_opportunities
                WHERE run_id = ? ORDER BY session, identity
                """,
                (run_id,),
            ).fetchall()
        return tuple(
            CohortOpportunity(
                run_id=str(row["run_id"]),
                session=date.fromisoformat(str(row["session"])),
                pre_move_m=float(row["pre_move_m"]),
            )
            for row in rows
        )

    def save_signals(self, signals: Sequence[StrategySignal], now: datetime) -> None:
        """Durably upsert Stage 6 signal state used by restart recovery."""

        rows = tuple(
            (
                signal.signal_id,
                signal.run_id,
                signal.session.isoformat(),
                signal.status.value,
                json.dumps(_signal_payload(signal), sort_keys=True),
                _aware(now).isoformat(),
            )
            for signal in signals
        )
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO runtime_signals
                (signal_id, run_id, session, status, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (signal_id) DO UPDATE SET
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                rows,
            )

    def load_signals(self, run_id: str) -> tuple[StrategySignal, ...]:
        """Load signals for one run; callers apply the live session/window rules."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM runtime_signals
                WHERE run_id = ? ORDER BY session, signal_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(_signal_from_payload(json.loads(str(row["payload"]))) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


class StockerRuntime:
    """One restartable application with per-run execution routing."""

    _SUPPORTED_STRATEGIES = {
        SESSION_HARD_METHOD.config_name,
        SESSION_HARD_METHOD.strategy_id,
        SESSION_HARD_HV_METHOD.config_name,
        SESSION_HARD_HV_METHOD.strategy_id,
    }

    def __init__(
        self,
        *,
        config: RunsConfig,
        ledger: ExecutionLedger,
        store: RuntimeStore,
        qualify: Qualifier,
        stage5: Stage5Analyzer,
        stage5_by_strategy: Mapping[str, Stage5Analyzer] | None = None,
        context_provider: StrategyContextProvider,
        entry_source: EntryBarSource,
        execution_router: ExecutionRouter | None = None,
        broker: RuntimeBroker | None = None,
        expected_account: str | None = None,
        session_resolver: SessionResolver | None = None,
        clock: Callable[[], datetime] | None = None,
        logger: Any | None = None,
        broker_sync_interval_seconds: float = 5.0,
        broker_factory: Callable[[IbkrConfig], RuntimeBroker] | None = None,
    ) -> None:
        if execution_router is None:
            if broker is None or expected_account is None or not expected_account.strip():
                raise ValueError("runtime requires an execution router or PAPER expected_account")
            execution_router = ExecutionRouter(
                (ExecutionDestination(Environment.PAPER, expected_account, broker),)
            )
        if broker_sync_interval_seconds <= 0.0:
            raise ValueError("broker sync interval must be positive")
        self._router = execution_router
        self._destinations = {
            environment: execution_router.for_environment(environment)
            for environment in execution_router.environments
        }
        preferred_environment = (
            Environment.PAPER
            if Environment.PAPER in self._destinations
            else next(iter(self._destinations))
        )
        self._market_data_broker = self._destinations[preferred_environment].broker
        self._ledger = ledger
        self._store = store
        self._qualify = qualify
        self._stage5_by_strategy = {
            SESSION_HARD_METHOD.strategy_version: stage5,
            **dict(stage5_by_strategy or {}),
        }
        self._context_provider = context_provider
        self._entry_source = entry_source
        self._session_resolver = session_resolver or ExchangeSessionResolver()
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self._logger = logger or configure_logging()
        self._broker_factory = broker_factory or (
            lambda config: IbkrConnection(config, execution_enabled=True)
        )
        self._broker_sync_interval = timedelta(seconds=broker_sync_interval_seconds)
        self._config = config
        self._manager = RunManager(UniverseCatalog(config.universes), config.runs)
        self._state = ApplicationState.STOPPED
        self._run_states: dict[str, RunRuntimeState] = {}
        self._run_reasons: dict[str, str] = {}
        self._sessions: dict[str, MarketSession] = {}
        self._activity_qualification_session: dict[str, date] = {}
        self._execution: dict[str, Stage7ExecutionService] = {}
        self._legacy_execution: dict[tuple[str, Environment], Stage7ExecutionService] = {}
        self._strategy_runtimes: dict[str, Stage7StrategyRuntime] = {}
        self._environment_ready: dict[Environment, bool] = {
            environment: False for environment in self._destinations
        }
        self._environment_reconciled: dict[Environment, bool] = {
            environment: False for environment in self._destinations
        }
        self._strategies: dict[str, SessionHardStructureDStrategy] = {}
        self._qualification = Stage5QualificationResult((), ())
        self._expected_move_prepared: set[tuple[str, date, datetime, tuple[int, ...]]] = set()
        self._run_ready_at: dict[str, datetime] = {}
        self._last_sync: datetime | None = None
        self._stopping = False
        self._cycle_lock = asyncio.Lock()

    async def start(self) -> RuntimeStatus:
        """Connect, verify, reconcile, prepare, and only then expose READY."""

        now = _aware(self._clock())
        self._state = ApplicationState.STARTING
        self._stopping = False
        self._execution.clear()
        self._legacy_execution.clear()
        self._strategy_runtimes.clear()
        self._strategies.clear()
        self._sessions.clear()
        self._run_ready_at.clear()
        self._qualification = Stage5QualificationResult((), ())
        self._expected_move_prepared.clear()
        for environment in self._environment_ready:
            self._environment_ready[environment] = False
            self._environment_reconciled[environment] = False
        self._store.recover_interrupted(now)
        self._logger.info("application_start")
        configured_runs: list[RunInstance] = []
        for instance in self._manager.list_runs():
            run = instance.config
            if not run.enabled:
                self._set_run(run.run_id, RunRuntimeState.DISABLED, "disabled by configuration")
            elif run.strategy not in self._SUPPORTED_STRATEGIES:
                self._set_run(run.run_id, RunRuntimeState.DEGRADED, "unsupported strategy")
            elif run.session is None:
                self._set_run(
                    run.run_id, RunRuntimeState.DEGRADED, "explicit market session required"
                )
            elif run.risk is None:
                self._set_run(run.run_id, RunRuntimeState.DEGRADED, "risk config is required")
            else:
                try:
                    destination = self._router.for_environment(run.execution_environment)
                except ExecutionEnvironmentUnavailableError:
                    self._set_run(
                        run.run_id,
                        RunRuntimeState.DEGRADED,
                        "EXECUTION_ENVIRONMENT_UNAVAILABLE",
                    )
                    continue
                self._set_run(run.run_id, RunRuntimeState.STARTING, "")
                active = self._manager.start_run(run.run_id)
                configured_runs.append(active)
                self._execution[run.run_id] = Stage7ExecutionService(
                    run=run,
                    expected_account=destination.expected_account,
                    broker=destination.broker,
                    ledger=self._ledger,
                    clock=self._clock,
                )

        if not configured_runs:
            if any(instance.config.enabled for instance in self._manager.list_runs()):
                self._state = ApplicationState.DEGRADED
            else:
                self._state = ApplicationState.READY
                self._logger.info(
                    "application_ready",
                    environments=0,
                    runs=0,
                    qualified=0,
                    ineligible=0,
                )
            return self.status()

        connected_environments: set[Environment] = set()
        for environment in dict.fromkeys(run.config.environment for run in configured_runs):
            destination = self._destinations[environment]
            affected = [run for run in configured_runs if run.config.environment is environment]
            try:
                session = await destination.broker.connect()
            except Exception as exc:
                self._degrade_runs(affected, f"broker unavailable: {exc}")
                self._logger.error(
                    "ibkr_connection_failed",
                    environment=environment.value,
                    reason=str(exc),
                )
                continue
            self._logger.info(
                "ibkr_connected",
                environment=session.environment.value,
                account=session.masked_account_id,
            )
            if not _session_matches_destination(session, destination):
                self._degrade_runs(affected, "account or environment mismatch")
                self._logger.error(
                    "account_verification_failed",
                    expected_environment=environment.value,
                    actual_environment=session.environment.value,
                    account=session.masked_account_id,
                )
                continue
            connected_environments.add(environment)
            self._logger.info(
                "account_verified",
                environment=environment.value,
                account=session.masked_account_id,
            )

        self._state = ApplicationState.RECONCILING
        runnable: list[RunInstance] = []
        for environment in connected_environments:
            destination = self._destinations[environment]
            affected = [run for run in configured_runs if run.config.environment is environment]
            services = tuple(
                (instance.config.run_id, self._execution[instance.config.run_id])
                for instance in affected
            )
            result = await self._reconcile_execution_services(services)
            self._store.record_reconciliation(
                environment=environment,
                account=destination.expected_account,
                connection_epoch=destination.broker.connection_epoch,
                ok=result.ok,
                detail=result.detail,
                now=now,
            )
            environment_ok = result.ok
            if not result.ok:
                for instance in affected:
                    self._set_run(instance.config.run_id, RunRuntimeState.DEGRADED, result.detail)
                    self._store.increment(
                        instance.config.run_id, now.date(), "reconciliation_issues"
                    )
                    self._logger.error(
                        "reconciliation_required",
                        environment=environment.value,
                        account=mask_ibkr_account(destination.expected_account),
                        run_id=instance.config.run_id,
                        reason=result.detail,
                    )
            self._environment_reconciled[environment] = environment_ok
            self._environment_ready[environment] = environment_ok
            if not environment_ok:
                continue
            for instance in affected:
                execution = self._execution[instance.config.run_id]
                self._ensure_strategy(instance.config, execution, now)
                market = self._resolve_market(instance, now)
                if market is None:
                    continue
                self._sessions[instance.config.run_id] = market
                self._set_run(
                    instance.config.run_id,
                    (
                        RunRuntimeState.ACTIVE
                        if market.state is MarketSessionState.ACTIVE_SESSION
                        else RunRuntimeState.READY
                    ),
                    "",
                )
                runnable.append(instance)
            self._logger.info(
                "reconciliation_complete",
                environment=environment.value,
                account=mask_ibkr_account(destination.expected_account),
                runs=len(affected),
            )

        runnable_runs = tuple(runnable)
        if not runnable_runs:
            self._state = ApplicationState.DEGRADED
            return self.status()
        try:
            self._qualification = await self._qualify(runnable_runs)
        except Exception as exc:
            self._degrade_runs(runnable_runs, f"instrument preparation failed: {exc}")
            self._state = ApplicationState.DEGRADED
            self._logger.error("instrument_preparation_failed", reason=str(exc))
            return self.status()
        _log_candidate_screen_failures(self._logger, self._qualification)
        self._remember_activity_qualification_sessions(runnable_runs)
        for environment in connected_environments:
            if self._environment_ready[environment]:
                await self._refresh_position_marks(
                    environment, self._destinations[environment], now
                )
        self._run_ready_at.update({instance.config.run_id: now for instance in runnable_runs})
        self._last_sync = now
        self._mark_missed_before(now)
        self._state = ApplicationState.READY
        self._logger.info(
            "application_ready",
            environments=len({run.config.environment for run in runnable_runs}),
            runs=len(runnable_runs),
            qualified=len(self._qualification.requests),
            ineligible=len(self._qualification.ineligible),
        )
        return self.status()

    async def stop(self) -> RuntimeStatus:
        """Stop new decisions, persist local state, and disconnect without cancelling protection."""

        self._stopping = True
        self._state = ApplicationState.STOPPING
        async with self._cycle_lock:
            for instance in self._manager.list_runs():
                if self._run_states.get(instance.config.run_id) is not RunRuntimeState.DISABLED:
                    self._set_run(instance.config.run_id, RunRuntimeState.STOPPED, "")
                    if instance.state.value == "ACTIVE":
                        self._manager.stop_run(instance.config.run_id)
            for destination in self._destinations.values():
                if destination.broker.is_connected:
                    destination.broker.disconnect()
                self._environment_ready[destination.environment] = False
                self._environment_reconciled[destination.environment] = False
            self._state = ApplicationState.STOPPED
            self._logger.info("application_stop")
        return self.status()

    @property
    def store(self) -> RuntimeStore:
        """Expose the durable operational store for diagnostics and queries."""

        return self._store

    async def apply_runs_config(
        self,
        config: RunsConfig,
        changed_run_ids: frozenset[str],
    ) -> RuntimeStatus:
        """Apply validated run changes under the scheduler's existing cycle lock."""

        if not changed_run_ids:
            return self.status()
        current_by_id = {item.run_id: item for item in self._config.runs}
        updated_by_id = {item.run_id: item for item in config.runs}
        current_universes = {item.universe_id: item for item in self._config.universes}
        updated_universes = {item.universe_id: item for item in config.universes}
        removed = set(current_by_id) - set(updated_by_id)
        if removed:
            raise ValueError("hot apply cannot delete run identities")
        unknown = changed_run_ids - set(updated_by_id)
        if unknown:
            raise ValueError(f"unknown changed runs: {', '.join(sorted(unknown))}")
        if any(
            current_by_id[run_id] != updated_by_id[run_id]
            for run_id in (set(current_by_id) & set(updated_by_id)) - changed_run_ids
        ):
            raise ValueError("hot apply payload changed an undeclared run")

        async with self._cycle_lock:
            now = _aware(self._clock())
            previous_states = dict(self._run_states)
            prepared_runs: list[RunInstance] = []
            previous_environments: dict[str, Environment] = {}
            self._config = config
            self._manager = RunManager(UniverseCatalog(config.universes), config.runs)
            for run_id, state in previous_states.items():
                if (
                    run_id in updated_by_id
                    and updated_by_id[run_id].enabled
                    and state
                    in {
                        RunRuntimeState.READY,
                        RunRuntimeState.ACTIVE,
                        RunRuntimeState.STARTING,
                    }
                ):
                    self._manager.start_run(run_id)

            for run_id in (item.run_id for item in config.runs if item.run_id in changed_run_ids):
                current = current_by_id.get(run_id)
                updated = updated_by_id[run_id]
                if not updated.enabled:
                    self._set_run(run_id, RunRuntimeState.DISABLED, "disabled by configuration")
                    self._replace_qualification_for({run_id}, Stage5QualificationResult((), ()))
                    self._activity_qualification_session.pop(run_id, None)
                    continue
                execution = self._execution.get(run_id)
                restart_required = current is None or any(
                    (
                        current.universe != updated.universe,
                        current_universes[current.universe] != updated_universes[updated.universe],
                        current.strategy != updated.strategy,
                        current.environment is not updated.environment,
                        current.session != updated.session,
                        not current.enabled,
                    )
                )
                if not restart_required and execution is not None:
                    execution.update_run_config(updated)
                    continue
                try:
                    destination = self._router.for_environment(updated.environment)
                except ExecutionEnvironmentUnavailableError:
                    self._set_run(
                        run_id,
                        RunRuntimeState.DEGRADED,
                        "EXECUTION_ENVIRONMENT_UNAVAILABLE",
                    )
                    continue
                self._set_run(run_id, RunRuntimeState.STARTING, "")
                self._manager.start_run(run_id)
                execution = Stage7ExecutionService(
                    run=updated,
                    expected_account=destination.expected_account,
                    broker=destination.broker,
                    ledger=self._ledger,
                    clock=self._clock,
                )
                if (
                    current is not None
                    and current.environment is not updated.environment
                    and run_id in self._execution
                ):
                    # A newly current service can also manage older exposure in its account.
                    self._legacy_execution.pop((run_id, updated.environment), None)
                    self._legacy_execution[(run_id, current.environment)] = self._execution[run_id]
                self._execution[run_id] = execution
                result = await execution.reconcile()
                if not result.ok:
                    self._set_run(run_id, RunRuntimeState.DEGRADED, result.detail)
                    continue
                strategy = self._strategies.get(run_id)
                if strategy is None:
                    self._ensure_strategy(updated, execution, now)
                else:
                    self._strategy_runtimes[run_id] = Stage7StrategyRuntime(
                        strategy=strategy,
                        execution=execution,
                    )
                instance = self._manager.get_run(run_id)
                market = self._resolve_market(instance, now)
                if market is None:
                    continue
                self._sessions[run_id] = market
                prepared_runs.append(instance)
                if current is not None:
                    previous_environments[run_id] = current.environment

            prepared_run_ids = {instance.config.run_id for instance in prepared_runs}
            if prepared_runs:
                try:
                    qualification = await self._qualify(tuple(prepared_runs))
                except Exception as exc:
                    self._replace_qualification_for(
                        prepared_run_ids, Stage5QualificationResult((), ())
                    )
                    for run_id in prepared_run_ids:
                        self._set_run(
                            run_id,
                            RunRuntimeState.DEGRADED,
                            f"instrument preparation failed: {exc}",
                        )
                else:
                    self._replace_qualification_for(prepared_run_ids, qualification)
                    self._remember_activity_qualification_sessions(prepared_runs)
                    for run_id in prepared_run_ids:
                        self._run_ready_at[run_id] = now
                    self._mark_missed_before(now, prepared_run_ids)
                    for instance in prepared_runs:
                        run_id = instance.config.run_id
                        market = self._sessions[run_id]
                        self._set_run(
                            run_id,
                            RunRuntimeState.ACTIVE
                            if market.state is MarketSessionState.ACTIVE_SESSION
                            else RunRuntimeState.READY,
                            "",
                        )
                        self._logger.info(
                            "run_config_applied",
                            run_id=run_id,
                            previous_environment=(
                                previous_environments[run_id].value
                                if run_id in previous_environments
                                else None
                            ),
                            environment=instance.config.environment.value,
                        )
            self._state = (
                ApplicationState.READY
                if any(self._environment_ready.values())
                else ApplicationState.DEGRADED
            )
            return self.status()

    async def replace_broker_config(self, config: IbkrConfig) -> RuntimeStatus:
        """Reconnect and reconcile only one edited execution environment."""

        if config.expected_account is None:
            raise ValueError(
                f"{config.environment.value} broker configuration requires expected_account"
            )
        async with self._cycle_lock:
            environment = config.environment
            current = self._destinations.get(environment)
            if current is None:
                broker = self._broker_factory(config)
            else:
                position_exposure = any(
                    item.environment is environment
                    for item in self._ledger.broker_position_snapshots()
                )
                order_exposure = any(
                    item.environment is environment
                    for item in self._ledger.broker_open_order_snapshots()
                )
                ledger_exposure = bool(
                    self._ledger.active_records(environment, current.expected_account)
                )
                if current.expected_account != config.expected_account and (
                    position_exposure or order_exposure or ledger_exposure
                ):
                    raise ValueError("cannot change expected account while broker exposure exists")
                broker = cast(RuntimeBroker, current.broker)
                broker.disconnect()
                reconfigure = getattr(broker, "reconfigure", None)
                if reconfigure is None:
                    raise ValueError("runtime broker does not support connection reconfiguration")
                reconfigure(config)
            replacement = ExecutionDestination(environment, config.expected_account, broker)
            destinations = {**self._destinations, environment: replacement}
            self._router = ExecutionRouter(tuple(destinations.values()))
            self._destinations = destinations
            self._environment_ready[environment] = False
            self._environment_reconciled[environment] = False
            for run_id, execution in self._execution_services():
                if execution.run_environment is not environment:
                    continue
                replacement_execution = Stage7ExecutionService(
                    run=execution.run_config,
                    expected_account=config.expected_account,
                    broker=broker,
                    ledger=self._ledger,
                    clock=self._clock,
                )
                legacy_key = (run_id, environment)
                if self._legacy_execution.get(legacy_key) is execution:
                    self._legacy_execution[legacy_key] = replacement_execution
                    continue
                self._execution[run_id] = replacement_execution
                strategy = self._strategies.get(run_id)
                if strategy is not None:
                    self._strategy_runtimes[run_id] = Stage7StrategyRuntime(
                        strategy=strategy,
                        execution=replacement_execution,
                    )
            return await self._reconnect(environment, disconnect_first=False)

    async def poll_once(self) -> RuntimeStatus:
        """Process currently due checkpoints and causal entry observations once."""

        if self._state is not ApplicationState.READY:
            return self.status()
        async with self._cycle_lock:
            return await self._poll_once()

    async def _poll_once(self) -> RuntimeStatus:
        """Process one cycle while graceful shutdown is excluded."""

        now = _aware(self._clock())
        if self._state is not ApplicationState.READY or self._stopping:
            return self.status()
        required_environments = {
            execution.run_environment for _run_id, execution in self._execution_services()
        }
        for environment in required_environments:
            if not self._destinations[environment].broker.is_connected:
                self._note_disconnect(environment, now)
        if not any(self._environment_ready.values()):
            return self.status()
        sync_due = self._last_sync is None or now - self._last_sync >= self._broker_sync_interval
        if sync_due and not await self._refresh_execution_state(now):
            return self.status()

        await self._refresh_activity_sessions(now)
        await self._refresh_scheduled_activity_shortlists(now)
        await self._prepare_upcoming_expected_moves(now)

        due: dict[tuple[date, datetime, int], list[RunInstance]] = {}
        for instance in self._manager.list_runs():
            run = instance.config
            if self._run_states.get(run.run_id) not in {
                RunRuntimeState.READY,
                RunRuntimeState.ACTIVE,
            }:
                continue
            market = self._resolve_market(instance, now)
            if market is None:
                continue
            self._sessions[run.run_id] = market
            self._set_run(
                run.run_id,
                (
                    RunRuntimeState.ACTIVE
                    if market.state is MarketSessionState.ACTIVE_SESSION
                    else RunRuntimeState.READY
                ),
                "",
            )
            for checkpoint, t0 in market.checkpoint_times():
                if t0 > now:
                    continue
                if self._store.checkpoint_state(run.run_id, market.session, t0) is not None:
                    continue
                ready_at = self._run_ready_at.get(run.run_id)
                missed = ready_at is None or t0 < ready_at or now >= t0 + timedelta(minutes=5)
                if missed:
                    self._store.mark_missed(run.run_id, market.session, t0, checkpoint, now)
                    self._logger.info(
                        "checkpoint_skipped_missed",
                        run_id=run.run_id,
                        session=market.session.isoformat(),
                        t0=t0.isoformat(),
                    )
                    continue
                if self._store.reserve_checkpoint(run.run_id, market.session, t0, checkpoint, now):
                    due.setdefault((market.session, t0, checkpoint), []).append(instance)

        for (session, t0, checkpoint), instances in sorted(due.items()):
            await self._evaluate_group(instances, session=session, t0=t0, checkpoint=checkpoint)
        await self._observe_entries(now)
        return self.status()

    async def _prepare_upcoming_expected_moves(self, now: datetime) -> None:
        self._expected_move_prepared = {
            key for key in self._expected_move_prepared if key[2] >= now
        }
        upcoming: dict[tuple[date, datetime], set[str]] = {}
        for instance in self._manager.list_runs():
            run_id = instance.config.run_id
            if self._run_states.get(run_id) not in {
                RunRuntimeState.READY,
                RunRuntimeState.ACTIVE,
            }:
                continue
            strategy = self._strategies.get(run_id)
            if (
                strategy is None
                or strategy.strategy_version != SESSION_HARD_HV_METHOD.strategy_version
            ):
                continue
            market = self._sessions.get(run_id) or self._resolve_market(instance, now)
            if market is None:
                continue
            for _checkpoint, t0 in market.checkpoint_times():
                if not now < t0 <= now + HV_EXPECTED_MOVE_PREFETCH_LEAD:
                    continue
                if self._store.checkpoint_state(run_id, market.session, t0) is not None:
                    continue
                upcoming.setdefault((market.session, t0), set()).add(run_id)

        stage5 = self._stage5_by_strategy.get(SESSION_HARD_HV_METHOD.strategy_version)
        if stage5 is None:
            return
        for (session, t0), run_ids in sorted(upcoming.items()):
            requests, _ineligible = self._qualification_for(run_ids)
            con_ids = tuple(sorted(request.instrument.con_id for request in requests))
            key = (SESSION_HARD_HV_METHOD.strategy_version, session, t0, con_ids)
            if key in self._expected_move_prepared:
                continue
            self._expected_move_prepared.add(key)
            await stage5.prepare_expected_moves(requests, session=session, t0=t0)

    async def _refresh_scheduled_activity_shortlists(self, now: datetime) -> None:
        """Qualify fixed-time activity screens once they become causally due."""

        pending_run_ids = {
            membership.run_id
            for item in self._qualification.ineligible
            if item.symbol == "ACTIVITY_SHORTLIST_V1"
            and item.reason in {"SCHEDULED", "ACTIVITY_SHORTLIST_NOT_READY"}
            for membership in item.memberships
        }
        if not pending_run_ids:
            return
        due: list[RunInstance] = []
        for instance in self._manager.list_runs():
            if instance.config.run_id not in pending_run_ids:
                continue
            market = self._resolve_market(instance, now)
            if market is None or len(market.active_bar_starts) <= 3:
                continue
            if market.active_bar_starts[3] <= now:
                due.append(instance)
        if not due:
            return
        qualification = await self._qualify(tuple(due))
        run_ids = {instance.config.run_id for instance in due}
        self._replace_qualification_for(run_ids, qualification)
        _log_candidate_screen_failures(self._logger, qualification)

    async def _refresh_activity_sessions(self, now: datetime) -> None:
        """Rotate generic activity qualification onto each new trading session."""

        due: list[RunInstance] = []
        for instance in self._manager.list_runs():
            run = instance.config
            if (
                not run.enabled
                or run.screen is None
                or run.screen.method is not CandidateScreen.ACTIVITY_SHORTLIST_V1
                or self._run_states.get(run.run_id)
                not in {RunRuntimeState.READY, RunRuntimeState.ACTIVE}
            ):
                continue
            market = self._resolve_market(instance, now)
            if market is None or not market.active_bar_starts:
                continue
            if self._activity_qualification_session.get(run.run_id) != market.session:
                due.append(instance)
                self._sessions[run.run_id] = market
        if not due:
            return
        qualification = await self._qualify(tuple(due))
        run_ids = {instance.config.run_id for instance in due}
        self._replace_qualification_for(run_ids, qualification)
        self._remember_activity_qualification_sessions(due)
        _log_candidate_screen_failures(self._logger, qualification)

    def _remember_activity_qualification_sessions(self, instances: Sequence[RunInstance]) -> None:
        for instance in instances:
            screen = instance.config.screen
            market = self._sessions.get(instance.config.run_id)
            if (
                screen is not None
                and screen.method is CandidateScreen.ACTIVITY_SHORTLIST_V1
                and market is not None
                and market.active_bar_starts
            ):
                self._activity_qualification_session[instance.config.run_id] = market.session

    async def reconnect(self, environment: Environment | None = None) -> RuntimeStatus:
        """Reconnect only affected environments, then verify and reconcile each one."""

        async with self._cycle_lock:
            return await self._reconnect(environment)

    async def _reconnect(
        self,
        environment: Environment | None = None,
        *,
        disconnect_first: bool = True,
    ) -> RuntimeStatus:
        """Internal reconnect path shared by recovery and serialized controls."""

        now = _aware(self._clock())
        required = {execution.run_environment for _run_id, execution in self._execution_services()}
        targets = (
            {environment}
            if environment is not None
            else {
                item
                for item in required
                if not self._environment_ready[item]
                or not self._destinations[item].broker.is_connected
            }
        )
        if not targets:
            return self.status()
        self._state = ApplicationState.RECONCILING
        for target in targets:
            destination = self._destinations[target]
            affected = [
                instance
                for instance in self._manager.list_runs()
                if instance.config.run_id in self._execution
                and instance.config.environment is target
                and instance.config.enabled
            ]
            services = tuple(
                (run_id, execution)
                for run_id, execution in self._execution_services()
                if execution.run_environment is target
            )
            if disconnect_first:
                destination.broker.disconnect()
            self._environment_ready[target] = False
            self._environment_reconciled[target] = False
            try:
                session = await destination.broker.connect()
            except Exception as exc:
                self._degrade_runs(affected, f"broker unavailable: {exc}")
                self._logger.error(
                    "ibkr_reconnect_failed",
                    environment=target.value,
                    reason=str(exc),
                )
                continue
            if not _session_matches_destination(session, destination):
                self._degrade_runs(affected, "account or environment mismatch")
                self._logger.error(
                    "account_verification_failed",
                    expected_environment=target.value,
                    actual_environment=session.environment.value,
                    account=session.masked_account_id,
                )
                continue

            reconciled = False
            try:
                result = await self._reconcile_execution_services(services)
                self._store.record_reconciliation(
                    environment=target,
                    account=destination.expected_account,
                    connection_epoch=destination.broker.connection_epoch,
                    ok=result.ok,
                    detail=result.detail,
                    now=now,
                )
            except Exception as exc:
                detail = f"broker reconciliation unavailable: {exc}"
                for run_id, _execution in services:
                    self._set_run(run_id, RunRuntimeState.DEGRADED, detail)
                    self._store.increment(run_id, now.date(), "reconciliation_issues")
                    self._logger.error(
                        "reconciliation_failed",
                        environment=target.value,
                        run_id=run_id,
                        reason=str(exc),
                    )
            else:
                reconciled = result.ok
                for run_id, execution in services:
                    if not result.ok:
                        self._set_run(run_id, RunRuntimeState.DEGRADED, result.detail)
                        self._store.increment(run_id, now.date(), "reconciliation_issues")
                        self._logger.error(
                            "reconciliation_required",
                            environment=target.value,
                            run_id=run_id,
                            reason=result.detail,
                        )
                    else:
                        current_execution = self._execution.get(run_id)
                        if current_execution is execution:
                            self._ensure_strategy(
                                self._manager.get_run(run_id).config,
                                execution,
                                now,
                            )
            self._environment_reconciled[target] = reconciled
            self._environment_ready[target] = reconciled
            if reconciled:
                self._logger.info(
                    "ibkr_reconnected",
                    environment=target.value,
                    account=session.masked_account_id,
                )

        recovered: list[RunInstance] = []
        for instance in self._manager.list_runs():
            if (
                instance.config.run_id not in self._execution
                or instance.config.environment not in targets
                or not self._environment_ready[instance.config.environment]
                or not instance.config.enabled
            ):
                continue
            market = self._resolve_market(instance, now)
            if market is None:
                continue
            self._sessions[instance.config.run_id] = market
            recovered.append(instance)
        if recovered:
            try:
                recovered_qualification = await self._qualify(recovered)
            except Exception as exc:
                self._degrade_runs(recovered, f"instrument preparation failed: {exc}")
                self._logger.error(
                    "instrument_preparation_failed",
                    environments=sorted(
                        {instance.config.environment.value for instance in recovered}
                    ),
                    reason=str(exc),
                )
                for target in {instance.config.environment for instance in recovered}:
                    self._environment_ready[target] = False
                self._replace_qualification_for(
                    {instance.config.run_id for instance in recovered},
                    Stage5QualificationResult((), ()),
                )
                recovered = []
            else:
                _log_candidate_screen_failures(self._logger, recovered_qualification)
                self._replace_qualification_for(
                    {instance.config.run_id for instance in recovered},
                    recovered_qualification,
                )
                self._remember_activity_qualification_sessions(recovered)
        self._run_ready_at.update({instance.config.run_id: now for instance in recovered})
        self._last_sync = now if recovered else self._last_sync
        if recovered:
            self._mark_missed_before(now, {instance.config.run_id for instance in recovered})
            for instance in recovered:
                market = self._sessions[instance.config.run_id]
                self._set_run(
                    instance.config.run_id,
                    (
                        RunRuntimeState.ACTIVE
                        if market.state is MarketSessionState.ACTIVE_SESSION
                        else RunRuntimeState.READY
                    ),
                    "",
                )
        self._state = (
            ApplicationState.READY
            if any(self._environment_ready.values())
            else ApplicationState.DEGRADED
        )
        return self.status()

    async def run_forever(self, *, poll_interval_seconds: float = 1.0) -> None:
        """Run the single evaluation loop with one reconnect attempt per interval."""

        if poll_interval_seconds <= 0.0:
            raise ValueError("poll interval must be positive")
        await self.start()
        while not self._stopping:
            if self._state is ApplicationState.READY:
                await self.poll_once()
            execution_services = self._execution_services()
            if execution_services and any(
                not self._environment_ready[environment]
                or not self._destinations[environment].broker.is_connected
                for environment in {
                    execution.run_environment for _run_id, execution in execution_services
                }
            ):
                await self.reconnect()
            await asyncio.sleep(poll_interval_seconds)

    async def market_data_check(self) -> CurrentQuote:
        """Read one qualified current quote for the explicit Stage 8 PAPER smoke path."""

        if self._state is not ApplicationState.READY:
            raise RuntimeError("runtime is not READY")
        request = next(iter(self._qualification.requests), None)
        if request is None:
            raise RuntimeError("no qualified instrument is available for market-data smoke")
        broker = self._market_data_broker
        if not broker.is_connected:
            broker = next(
                (
                    destination.broker
                    for destination in self._destinations.values()
                    if destination.broker.is_connected
                ),
                broker,
            )
        current_quote = getattr(broker, "current_quote", None)
        if current_quote is None:
            raise RuntimeError("runtime broker does not expose market data")
        quote: CurrentQuote = await current_quote(request.instrument)
        return quote

    async def activity_scanner_readiness(self) -> dict[str, str]:
        """Report catalogue readiness from the connected scanner vocabulary."""

        broker = self._market_data_broker
        if not broker.is_connected:
            broker = next(
                (
                    destination.broker
                    for environment, destination in self._destinations.items()
                    if self._environment_ready.get(environment) and destination.broker.is_connected
                ),
                broker,
            )
        if not broker.is_connected:
            return {market.market_id.value: "BROKER_NOT_CONNECTED" for market in MARKET_CATALOGUE}
        discover = getattr(broker, "scanner_capabilities", None)
        if discover is None:
            return {market.market_id.value: "SCANNER_NOT_AVAILABLE" for market in MARKET_CATALOGUE}
        try:
            capabilities = await discover()
        except Exception as exc:
            message = str(exc).lower()
            status = (
                "DATA_NOT_ENTITLED"
                if any(
                    word in message
                    for word in ("entitle", "subscription", "market data permission")
                )
                else "SCANNER_NOT_AVAILABLE"
            )
            return {market.market_id.value: status for market in MARKET_CATALOGUE}
        run_markets = {
            run.run_id: run.market_id for run in self._config.runs if run.market_id is not None
        }
        qualified_markets = {
            run_markets[membership.run_id]
            for request in self._qualification.requests
            for membership in request.memberships
            if membership.run_id in run_markets
        }
        failed_markets = {
            run_markets[membership.run_id]
            for failure in self._qualification.ineligible
            if failure.symbol not in {"HOT_BY_VOLUME", "ACTIVITY_SHORTLIST_V1"}
            for membership in failure.memberships
            if membership.run_id in run_markets
        }
        return {
            market.market_id.value: (
                "CONTRACT_QUALIFICATION_FAILED"
                if market.market_id in failed_markets and market.market_id not in qualified_markets
                else "AVAILABLE"
                if market.scanner_location in capabilities.locations
                and capabilities.supports_instrument(
                    market.scanner_location, market.scanner_instrument
                )
                and len(
                    {component.value for component in ActivityScanner}
                    & set(capabilities.scan_codes_for(market.scanner_location))
                )
                >= 2
                else "SCANNER_NOT_AVAILABLE"
            )
            for market in MARKET_CATALOGUE
        }

    async def execution_readiness_diagnostic(
        self, environment: Environment
    ) -> ExecutionReadinessDiagnostic:
        """Read one environment's execution readiness without transmitting an order."""

        destination = self._router.for_environment(environment)
        broker = destination.broker
        connected = broker.is_connected
        account = broker.account or None
        account_match = (
            connected
            and broker.environment is environment
            and account == destination.expected_account
        )
        account_state_available = False
        open_order_count: int | None = None
        position_count: int | None = None
        detail = "READY"
        if not connected:
            detail = "BROKER_DISCONNECTED"
        elif not account_match:
            detail = "ACCOUNT_OR_ENVIRONMENT_MISMATCH"
        else:
            try:
                state = await broker.account_state()
                account_state_available = (
                    state.connected
                    and state.environment is environment
                    and state.account == destination.expected_account
                    and state.equity is not None
                )
                open_order_count = len(await broker.read_open_orders())
                position_count = len(await broker.read_positions())
            except Exception as exc:
                detail = f"broker state unavailable: {exc}"
            else:
                if not account_state_available:
                    detail = "ACCOUNT_OR_ENVIRONMENT_MISMATCH"
                elif not self._environment_reconciled[environment]:
                    detail = "EXECUTION_RECONCILIATION_REQUIRED"
        ready = (
            connected
            and account_match
            and account_state_available
            and open_order_count is not None
            and position_count is not None
            and self._environment_reconciled[environment]
            and self._environment_ready[environment]
        )
        if not ready and detail == "READY":
            detail = "EXECUTION_RECONCILIATION_REQUIRED"
        return ExecutionReadinessDiagnostic(
            environment=environment,
            connected=connected,
            account=account,
            expected_account=destination.expected_account,
            expected_account_match=account_match,
            account_state_available=account_state_available,
            open_orders=open_order_count,
            positions=position_count,
            reconciled=self._environment_reconciled[environment],
            ready=ready,
            detail=detail,
        )

    def record_fill(self, fill: BrokerFill) -> bool:
        """Persist one broker fill idempotently and expose its run-level operational count."""

        matching = None
        for record in self._ledger.active_records(fill.environment, fill.account):
            if fill.order_id in {
                record.parent_order_id,
                record.stop_order_id,
                record.target_order_id,
            }:
                matching = record
                break
        if matching is None:
            for run_id, execution in self._execution.items():
                if execution.run_environment is not fill.environment:
                    continue
                execution.record_fill(fill)
                self._set_run(
                    run_id,
                    RunRuntimeState.DEGRADED,
                    "EXECUTION_RECONCILIATION_REQUIRED: unexpected broker fill",
                )
            if fill.environment in self._environment_ready:
                self._environment_ready[fill.environment] = False
                self._environment_reconciled[fill.environment] = False
            self._state = (
                ApplicationState.READY
                if any(self._environment_ready.values())
                else ApplicationState.DEGRADED
            )
            return False
        matching_execution = self._execution.get(matching.run_id)
        if matching_execution is None or not matching_execution.record_fill(fill):
            return False
        market = self._sessions.get(matching.run_id)
        session = market.session if market is not None else _aware(fill.executed_at).date()
        self._store.increment(matching.run_id, session, "fills")
        updated = self._ledger.get(matching.order_plan_id)
        event = (
            "position_closed"
            if updated is not None and updated.status is OrderLifecycle.CLOSED
            else "position_opened"
        )
        self._logger.info(
            event,
            environment=fill.environment.value,
            account=mask_ibkr_account(fill.account),
            run_id=matching.run_id,
            signal_id=matching.signal_id,
            order_plan_id=matching.order_plan_id,
            conId=matching.con_id,
            ibkr_order_id=fill.order_id,
            execution_id=fill.execution_id,
        )
        return True

    def status(self) -> RuntimeStatus:
        now = _aware(self._clock())
        run_statuses: list[RunStatus] = []
        for instance in self._manager.list_runs():
            run = instance.config
            market = self._sessions.get(run.run_id)
            run_counters = self._store.counters(
                run.run_id, market.session if market is not None else now.date()
            )
            ready = sum(
                run.run_id in request_run_ids
                for request_run_ids in (
                    {membership.run_id for membership in request.memberships}
                    for request in self._qualification.requests
                )
            )
            run_statuses.append(
                RunStatus(
                    run_id=run.run_id,
                    universe=run.universe,
                    strategy=run.strategy,
                    environment=run.environment,
                    state=self._run_states.get(run.run_id, RunRuntimeState.STOPPED),
                    reason=self._run_reasons.get(run.run_id, ""),
                    session=market.session if market else None,
                    market=market.state if market else None,
                    instruments_ready=ready,
                    signals_today=run_counters.signals,
                    open_positions=sum(
                        record.filled_quantity > record.closed_quantity
                        for record in self._ledger.list_records(
                            run_id=run.run_id,
                            limit=500,
                        )[0]
                    ),
                )
            )
        execution_statuses = []
        for environment, destination in self._destinations.items():
            account_state = self._account_state_for(environment)
            execution_statuses.append(
                ExecutionEnvironmentStatus(
                    environment=environment,
                    connected=destination.broker.is_connected,
                    account=destination.broker.account or None,
                    expected_account=destination.expected_account,
                    reconciled=self._environment_reconciled[environment],
                    ready=self._environment_ready[environment],
                    equity=account_state.equity if account_state is not None else None,
                    buying_power=(
                        account_state.buying_power if account_state is not None else None
                    ),
                )
            )
        resource_status = getattr(self._market_data_broker, "resource_status", None)
        return RuntimeStatus(
            application=self._state,
            execution_environments=tuple(execution_statuses),
            runs=tuple(run_statuses),
            counters=self._store.counters(),
            ibkr_resources=resource_status() if resource_status is not None else None,
        )

    def _account_state_for(self, environment: Environment) -> BrokerAccountState | None:
        destination = self._destinations[environment]
        if not destination.broker.is_connected:
            return None
        for _run_id, execution in self._execution_services():
            if execution.run_environment is not environment:
                continue
            state = execution.last_account_state
            if (
                state is not None
                and state.connected
                and state.environment is environment
                and state.account == destination.expected_account
            ):
                return state
        return None

    def _set_run(self, run_id: str, state: RunRuntimeState, reason: str) -> None:
        self._run_states[run_id] = state
        self._run_reasons[run_id] = reason

    def _degrade_runs(self, runs: Sequence[RunInstance], reason: str) -> None:
        for instance in runs:
            self._set_run(instance.config.run_id, RunRuntimeState.DEGRADED, reason)

    def _ensure_strategy(
        self, run: RunConfig, execution: Stage7ExecutionService, now: datetime
    ) -> None:
        run_id = run.run_id
        if run_id in self._strategies:
            return
        default_method = (
            SESSION_HARD_HV_METHOD
            if run.strategy
            in {SESSION_HARD_HV_METHOD.config_name, SESSION_HARD_HV_METHOD.strategy_id}
            else SESSION_HARD_METHOD
        )
        strategy_id = str(run.strategy_id or default_method.strategy_id)
        strategy_version = str(run.strategy_version or default_method.strategy_version)
        strategy = create_strategy(strategy_id, strategy_version)
        restored = self._store.load_signals(run_id)
        strategy.restore_signals(restored)
        expired = strategy.expire_waiting_before(now)
        if expired:
            self._store.save_signals(expired, now)
            self._logger.info("signals_expired_during_recovery", run_id=run_id, count=len(expired))
        self._strategies[run_id] = strategy
        self._strategy_runtimes[run_id] = Stage7StrategyRuntime(
            strategy=strategy, execution=execution
        )

    def _resolve_market(self, instance: RunInstance, now: datetime) -> MarketSession | None:
        try:
            return self._session_resolver.resolve(instance.config, now)
        except Exception as exc:
            self._set_run(
                instance.config.run_id,
                RunRuntimeState.DEGRADED,
                f"invalid market session: {exc}",
            )
            self._logger.error(
                "run_degraded",
                run_id=instance.config.run_id,
                reason=f"invalid market session: {exc}",
            )
            return None

    def _mark_missed_before(self, ready_at: datetime, run_ids: set[str] | None = None) -> None:
        for instance in self._manager.list_runs():
            run_id = instance.config.run_id
            if run_ids is not None and run_id not in run_ids:
                continue
            market = self._sessions.get(run_id)
            if market is None:
                continue
            for checkpoint, t0 in market.checkpoint_times():
                if t0 < ready_at:
                    self._store.mark_missed(run_id, market.session, t0, checkpoint, ready_at)

    async def _evaluate_group(
        self,
        instances: Sequence[RunInstance],
        *,
        session: date,
        t0: datetime,
        checkpoint: int,
    ) -> None:
        grouped: dict[str, list[RunInstance]] = {}
        for instance in instances:
            strategy = self._strategies.get(instance.config.run_id)
            if strategy is None:
                self._fail_checkpoint(
                    instance,
                    session,
                    t0,
                    "Runtime strategy is not initialized",
                    _aware(self._clock()),
                )
                continue
            version = strategy.strategy_version
            grouped.setdefault(version, []).append(instance)
        for version in sorted(grouped):
            stage5 = self._stage5_by_strategy.get(version)
            if stage5 is None:
                now = _aware(self._clock())
                for instance in grouped[version]:
                    self._fail_checkpoint(
                        instance,
                        session,
                        t0,
                        f"No Stage 5 service configured for {version}",
                        now,
                    )
                continue
            await self._evaluate_strategy_group(
                grouped[version],
                stage5=stage5,
                session=session,
                t0=t0,
                checkpoint=checkpoint,
            )

    async def _evaluate_strategy_group(
        self,
        instances: Sequence[RunInstance],
        *,
        stage5: Stage5Analyzer,
        session: date,
        t0: datetime,
        checkpoint: int,
    ) -> None:
        now = _aware(self._clock())
        run_ids = {instance.config.run_id for instance in instances}
        requests, ineligible = self._qualification_for(run_ids)
        try:
            rows = await stage5.analyze(
                requests,
                ineligible=ineligible,
                session=session,
                t0=t0,
            )
        except Exception as exc:
            for instance in instances:
                self._fail_checkpoint(instance, session, t0, str(exc), now)
            return

        for instance in instances:
            run = instance.config
            run_rows = tuple(row for row in rows if run.run_id in row.run_ids)
            valid_rows = tuple(
                row
                for row in run_rows
                if row.session == session and row.t0 == t0 and row.t0.tzinfo is not None
            )
            if len(valid_rows) != len(run_rows):
                self._fail_checkpoint(
                    instance,
                    session,
                    t0,
                    "STALE_OR_SESSION_MISMATCHED_INPUT",
                    now,
                )
                continue
            for row in valid_rows:
                if row.status is Stage5Status.READY:
                    continue
                self._logger.warning(
                    "stage5_candidate_rejected",
                    run_id=run.run_id,
                    con_id=row.con_id,
                    symbol=row.symbol,
                    status=row.status.value,
                    reason=row.exclusion_reason,
                )
            try:
                instruments = {
                    request.instrument.con_id: request.instrument for request in requests
                }
                context = await self._context_provider.context_for(
                    run,
                    valid_rows,
                    checkpoint,
                    instruments,
                    self._store.cohort_history(run.run_id),
                )
                if context.run_id != run.run_id:
                    raise ValueError("strategy context run identity mismatch")
                strategy = self._strategies[run.run_id]
            except Exception as exc:
                self._fail_checkpoint(instance, session, t0, str(exc), now)
                continue
            evaluated: list[StrategySignal] = []
            candidate_errors = 0
            for row in valid_rows:
                try:
                    evaluated.extend(strategy.evaluate((row,), context))
                except Exception as exc:
                    candidate_errors += 1
                    self._logger.error(
                        "strategy_candidate_failed",
                        run_id=run.run_id,
                        con_id=row.con_id,
                        symbol=row.symbol,
                        reason=str(exc),
                    )
            self._store.save_signals(evaluated, now)
            waiting = sum(signal.status is SignalStatus.WAITING_FOR_ENTRY for signal in evaluated)
            self._store.increment(run.run_id, session, "signals", waiting)
            ready_count = sum(row.status is Stage5Status.READY for row in valid_rows)
            context_not_ready = sum(
                row.status is Stage5Status.PRE_CONTEXT_NOT_READY for row in valid_rows
            )
            self._store.increment(run.run_id, session, "checkpoints_processed")
            self._store.increment(run.run_id, session, "instruments_ready", ready_count)
            self._store.mark_checkpoint(
                run.run_id,
                session,
                t0,
                CheckpointState.COMPLETED,
                f"Stage 5 ready={ready_count}; candidate_errors={candidate_errors}",
                now,
            )
            self._logger.info(
                "checkpoint_processed",
                run_id=run.run_id,
                session=session.isoformat(),
                t0=t0.isoformat(),
                checkpoint=checkpoint,
                instruments_ready=ready_count,
                pre_context_not_ready=context_not_ready,
                strategy_signals=waiting,
                candidate_errors=candidate_errors,
            )

    async def _observe_entries(self, now: datetime) -> None:
        for instance in self._manager.list_runs():
            run = instance.config
            if self._run_states.get(run.run_id) not in {
                RunRuntimeState.READY,
                RunRuntimeState.ACTIVE,
            }:
                continue
            market = self._sessions.get(run.run_id)
            strategy_runtime = self._strategy_runtimes.get(run.run_id)
            if market is None or strategy_runtime is None:
                continue
            requests, _ineligible = self._qualification_for({run.run_id})
            instruments = {request.instrument.con_id: request.instrument for request in requests}
            try:
                strategy = self._strategies[run.run_id]
                expired = strategy.expire_waiting_before(now)
                if expired:
                    self._store.save_signals(expired, now)
                bars = await self._entry_source.bars_for(
                    run,
                    instruments,
                    session=market.session,
                    now=now,
                    signals=strategy.signals,
                )
                attempts = await strategy_runtime.observe_and_execute(bars, instruments)
                self._store.save_signals(strategy.signals, now)
            except Exception as exc:
                self._set_run(run.run_id, RunRuntimeState.DEGRADED, str(exc))
                self._logger.error("run_degraded", run_id=run.run_id, reason=str(exc))
                continue
            for attempt in attempts:
                self._logger.info(
                    "strategy_signal",
                    environment=attempt.environment.value,
                    account=mask_ibkr_account(attempt.actual_account),
                    run_id=run.run_id,
                    signal_id=attempt.signal_id,
                    order_plan_id=(
                        attempt.order_plan.order_plan_id if attempt.order_plan else None
                    ),
                    conId=attempt.order_plan.con_id if attempt.order_plan else None,
                    result=attempt.code.value,
                )
                if attempt.code is ExecutionResultCode.SUBMITTED:
                    self._store.increment(run.run_id, market.session, "orders")
                    self._logger.info(
                        "order_submitted",
                        environment=attempt.environment.value,
                        account=mask_ibkr_account(attempt.actual_account),
                        run_id=run.run_id,
                        signal_id=attempt.signal_id,
                        order_plan_id=(
                            attempt.order_plan.order_plan_id if attempt.order_plan else None
                        ),
                        conId=attempt.order_plan.con_id if attempt.order_plan else None,
                        ibkr_order_id=attempt.order_ids.parent if attempt.order_ids else None,
                    )
                elif attempt.code is ExecutionResultCode.BROKER_REJECTED:
                    self._store.increment(run.run_id, market.session, "broker_rejects")
                    self._logger.error(
                        "broker_order_rejected",
                        environment=attempt.environment.value,
                        account=mask_ibkr_account(attempt.actual_account),
                        run_id=run.run_id,
                        signal_id=attempt.signal_id,
                        order_plan_id=(
                            attempt.order_plan.order_plan_id if attempt.order_plan else None
                        ),
                        conId=attempt.order_plan.con_id if attempt.order_plan else None,
                        result=attempt.code.value,
                        reason=attempt.detail,
                    )
                elif attempt.code is not ExecutionResultCode.DUPLICATE_ORDER_BLOCKED:
                    self._store.increment(run.run_id, market.session, "risk_rejects")
                    self._logger.warning(
                        "execution_rejected",
                        environment=attempt.environment.value,
                        account=mask_ibkr_account(attempt.actual_account),
                        run_id=run.run_id,
                        signal_id=attempt.signal_id,
                        order_plan_id=(
                            attempt.order_plan.order_plan_id if attempt.order_plan else None
                        ),
                        conId=attempt.order_plan.con_id if attempt.order_plan else None,
                        result=attempt.code.value,
                        reason=attempt.detail,
                    )
            for index, opportunity in enumerate(strategy.cohort_opportunities):
                identity = (
                    f"{opportunity.run_id}|{opportunity.session.isoformat()}|"
                    f"{opportunity.pre_move_m:.17g}|{index}"
                )
                self._store.save_cohort(identity, opportunity)

    def _qualification_for(
        self, run_ids: set[str]
    ) -> tuple[tuple[Stage5QualifiedRequest, ...], tuple[Stage5IneligibleInstrument, ...]]:
        requests: list[Stage5QualifiedRequest] = []
        for request in self._qualification.requests:
            memberships = tuple(
                membership for membership in request.memberships if membership.run_id in run_ids
            )
            if memberships:
                requests.append(Stage5QualifiedRequest(request.instrument, memberships))
        ineligible: list[Stage5IneligibleInstrument] = []
        for item in self._qualification.ineligible:
            memberships = tuple(
                membership for membership in item.memberships if membership.run_id in run_ids
            )
            if memberships:
                ineligible.append(
                    Stage5IneligibleInstrument(item.symbol, memberships, item.reason, item.status)
                )
        return tuple(requests), tuple(ineligible)

    def _replace_qualification_for(
        self,
        run_ids: set[str],
        replacement: Stage5QualificationResult,
    ) -> None:
        retained_run_ids = set(self._execution) - run_ids
        retained_requests, retained_ineligible = self._qualification_for(retained_run_ids)
        self._qualification = _merge_qualification_results(
            Stage5QualificationResult(retained_requests, retained_ineligible),
            replacement,
        )

    def _fail_checkpoint(
        self,
        instance: RunInstance,
        session: date,
        t0: datetime,
        detail: str,
        now: datetime,
    ) -> None:
        run_id = instance.config.run_id
        self._store.mark_checkpoint(run_id, session, t0, CheckpointState.FAILED, detail, now)
        self._set_run(run_id, RunRuntimeState.DEGRADED, detail)
        self._logger.error("run_degraded", run_id=run_id, reason=detail)

    def _note_disconnect(self, environment: Environment, now: datetime) -> None:
        self._environment_ready[environment] = False
        self._environment_reconciled[environment] = False
        for run_id, execution in self._execution_services():
            if execution.run_environment is not environment:
                continue
            self._set_run(run_id, RunRuntimeState.DEGRADED, "BROKER_DISCONNECTED")
            self._store.increment(run_id, now.date(), "disconnects")
        self._state = (
            ApplicationState.READY
            if any(self._environment_ready.values())
            else ApplicationState.DEGRADED
        )
        self._logger.error("ibkr_disconnected", environment=environment.value)

    async def _reconcile_execution_services(
        self,
        services: Sequence[tuple[str, Stage7ExecutionService]],
    ) -> ReconciliationResult:
        """Read one account-wide broker snapshot and share it with sibling runs."""

        if not services:
            raise ValueError("at least one execution service is required for reconciliation")
        source = services[0][1]
        try:
            result = await source.reconcile()
            if result.ok:
                for _run_id, execution in services[1:]:
                    execution.adopt_reconciliation(source)
            else:
                for _run_id, execution in services[1:]:
                    execution.invalidate_reconciliation()
            return result
        except Exception:
            for _run_id, execution in services:
                execution.invalidate_reconciliation()
            raise

    async def _refresh_execution_state(self, now: datetime) -> bool:
        """Poll durable broker state without allowing orders during the refresh."""

        self._state = ApplicationState.RECONCILING
        for environment in {
            execution.run_environment for _run_id, execution in self._execution_services()
        }:
            destination = self._destinations[environment]
            if not destination.broker.is_connected:
                self._note_disconnect(environment, now)
                continue
            services = tuple(
                (run_id, execution)
                for run_id, execution in self._execution_services()
                if execution.run_environment is environment
            )
            try:
                result = await self._reconcile_execution_services(services)
            except Exception as exc:
                detail = f"broker reconciliation unavailable: {exc}"
                for run_id, _execution in services:
                    self._set_run(run_id, RunRuntimeState.DEGRADED, detail)
                    self._store.increment(run_id, now.date(), "reconciliation_issues")
                    self._logger.error(
                        "reconciliation_failed",
                        environment=environment.value,
                        run_id=run_id,
                        reason=str(exc),
                    )
                environment_ok = False
            else:
                self._store.record_reconciliation(
                    environment=environment,
                    account=destination.expected_account,
                    connection_epoch=destination.broker.connection_epoch,
                    ok=result.ok,
                    detail=result.detail,
                    now=now,
                )
                environment_ok = result.ok
                for run_id, execution in services:
                    if not result.ok:
                        self._set_run(run_id, RunRuntimeState.DEGRADED, result.detail)
                        self._store.increment(run_id, now.date(), "reconciliation_issues")
                        self._logger.error(
                            "reconciliation_required",
                            environment=environment.value,
                            run_id=run_id,
                            reason=result.detail,
                        )
                    else:
                        self._ensure_strategy(self._manager.get_run(run_id).config, execution, now)
            self._environment_reconciled[environment] = environment_ok
            self._environment_ready[environment] = environment_ok
            if environment_ok:
                await self._refresh_position_marks(environment, destination, now)
        self._last_sync = now
        self._state = (
            ApplicationState.READY
            if any(self._environment_ready.values())
            else ApplicationState.DEGRADED
        )
        return self._state is ApplicationState.READY

    async def _refresh_position_marks(
        self,
        environment: Environment,
        destination: ExecutionDestination,
        now: datetime,
    ) -> None:
        current_quote = getattr(destination.broker, "current_quote", None)
        if current_quote is None:
            return
        instruments = {
            request.instrument.con_id: request.instrument
            for request in self._qualification.requests
        }
        active_records = self._ledger.active_records(environment, destination.expected_account)
        run_instances = {instance.config.run_id: instance for instance in self._manager.list_runs()}
        positions = (
            item
            for item in self._ledger.broker_position_snapshots()
            if item.environment is environment and item.account == destination.expected_account
        )
        for position in positions:
            instrument = instruments.get(position.con_id)
            if instrument is None:
                record = next(
                    (
                        item
                        for item in active_records
                        if item.con_id == position.con_id
                        and item.filled_quantity > item.closed_quantity
                    ),
                    None,
                )
                instance = run_instances.get(record.run_id) if record is not None else None
                if instance is None:
                    continue
                member = next(
                    (item for item in instance.universe.members if item.symbol == position.symbol),
                    None,
                )
                market = (
                    get_market(instance.config.market_id)
                    if instance.config.market_id is not None
                    else None
                )
                currency = (
                    market.currency if market is not None else member.currency if member else ""
                )
                if not currency:
                    continue
                instrument = QualifiedInstrument(
                    symbol=position.symbol,
                    con_id=position.con_id,
                    exchange=member.exchange if member is not None else "SMART",
                    primary_exchange=(member.primary_exchange if member is not None else None),
                    currency=currency,
                    security_type=market.security_type if market is not None else "STK",
                )
            try:
                quote: CurrentQuote = await current_quote(instrument)
            except Exception as exc:
                self._logger.warning(
                    "position_mark_unavailable",
                    environment=environment.value,
                    con_id=position.con_id,
                    reason=str(exc),
                )
                continue
            mark = _quote_mark(quote)
            if mark is None:
                continue
            self._ledger.record_position_mark(
                environment=environment,
                account=destination.expected_account,
                con_id=position.con_id,
                mark=mark,
                observed_at=now,
            )

    def _execution_services(self) -> tuple[tuple[str, Stage7ExecutionService], ...]:
        return (
            *self._execution.items(),
            *(
                (run_id, service)
                for (run_id, _environment), service in self._legacy_execution.items()
            ),
        )


def _session_matches_destination(session: BrokerSession, destination: ExecutionDestination) -> bool:
    return (
        session.environment is destination.environment
        and session.account_id == destination.expected_account
        and destination.broker.environment is destination.environment
        and destination.broker.account == destination.expected_account
    )


def _quote_mark(quote: CurrentQuote) -> float | None:
    if quote.last is not None and quote.last > 0:
        return quote.last
    if quote.bid is not None and quote.ask is not None and quote.bid > 0 and quote.ask >= quote.bid:
        return (quote.bid + quote.ask) / 2
    if quote.close is not None and quote.close > 0:
        return quote.close
    return None


def _log_candidate_screen_failures(logger: Any, result: Stage5QualificationResult) -> None:
    for failure in result.ineligible:
        if failure.symbol not in {"HOT_BY_VOLUME", "ACTIVITY_SHORTLIST_V1"}:
            continue
        logger.warning(
            "candidate_screen_ineligible",
            runs=[membership.run_id for membership in failure.memberships],
            universes=[membership.universe_id for membership in failure.memberships],
            reason=failure.reason,
        )


def _merge_qualification_results(
    *results: Stage5QualificationResult,
) -> Stage5QualificationResult:
    qualified: dict[int, tuple[QualifiedInstrument, set[Stage5Membership]]] = {}
    ineligible: list[Stage5IneligibleInstrument] = []
    for result in results:
        ineligible.extend(result.ineligible)
        for request in result.requests:
            existing = qualified.get(request.instrument.con_id)
            if existing is None:
                qualified[request.instrument.con_id] = (
                    request.instrument,
                    set(request.memberships),
                )
            else:
                existing[1].update(request.memberships)
    requests = tuple(
        Stage5QualifiedRequest(
            instrument,
            tuple(sorted(memberships, key=lambda item: (item.universe_id, item.run_id))),
        )
        for _con_id, (instrument, memberships) in sorted(qualified.items())
    )
    return Stage5QualificationResult(requests, tuple(ineligible))


def _signal_payload(signal: StrategySignal) -> dict[str, object]:
    payload: dict[str, object] = asdict(signal)
    payload["session"] = signal.session.isoformat()
    payload["t0"] = _aware(signal.t0).isoformat()
    payload["status"] = signal.status.value
    payload["band"] = signal.band.value if signal.band is not None else None
    for name in ("entry_timestamp", "signal_timestamp"):
        value = getattr(signal, name)
        payload[name] = _aware(value).isoformat() if value is not None else None
    return payload


def _signal_from_payload(payload: Mapping[str, object]) -> StrategySignal:
    values = dict(payload)
    values["session"] = date.fromisoformat(str(values["session"]))
    values["t0"] = datetime.fromisoformat(str(values["t0"]))
    values["status"] = SignalStatus(str(values["status"]))
    values["band"] = PreMoveBand(str(values["band"])) if values.get("band") else None
    for name in ("entry_timestamp", "signal_timestamp"):
        if values.get(name) is not None:
            values[name] = datetime.fromisoformat(str(values[name]))
    return StrategySignal(**values)  # type: ignore[arg-type]


def _optional_schedule_time(value: object) -> datetime | None:
    if value is None or str(value) == "NaT":
        return None
    to_datetime = getattr(value, "to_pydatetime", None)
    if to_datetime is None:
        return None
    return _aware(to_datetime())


def _five_minute_slots(start: datetime, end: datetime) -> tuple[datetime, ...]:
    cursor = _aware(start)
    finish = _aware(end)
    values: list[datetime] = []
    while cursor < finish:
        values.append(cursor)
        cursor += timedelta(minutes=5)
    return tuple(values)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("runtime timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _intraday_timestamp(value: date | datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("runtime entry bars require intraday timestamps")
    return _aware(value)


class IbkrSessionDataSource:
    """Supply Stage 6 causal score inputs and entry bars from the shared IBKR cache."""

    _FIVE_MINUTES = HistorySemantics("5 mins", "TRADES", True)
    _ONE_MINUTE = HistorySemantics("1 min", "TRADES", True)

    def __init__(
        self,
        ibkr: IbkrConnection,
        history_cache: IbkrHistoryCache,
        *,
        logger: Any | None = None,
    ) -> None:
        self._cache = history_cache
        self._history = IbkrHistoryService(ibkr, history_cache)
        self._logger = logger or configure_logging()

    async def context_for(
        self,
        run: RunConfig,
        rows: Sequence[Stage5FeatureSnapshot],
        checkpoint: int,
        instruments: Mapping[int, QualifiedInstrument],
        cohort_history: Sequence[CohortOpportunity],
    ) -> StrategyContext:
        assessments: dict[StrategyOpportunityKey, SessionHardAssessment] = {}
        for row in rows:
            if row.status is not Stage5Status.READY or row.con_id is None:
                continue
            instrument = instruments.get(row.con_id)
            if instrument is None:
                self._logger.warning(
                    "session_hard_input_unavailable",
                    run_id=run.run_id,
                    con_id=row.con_id,
                    symbol=row.symbol,
                    reason="qualified instrument identity missing",
                )
                continue
            t0 = _aware(row.t0)
            market_session = ExchangeSessionResolver().resolve(run, t0)
            required = market_session.active_bar_starts[:checkpoint]
            if len(required) != checkpoint:
                self._logger.warning(
                    "session_hard_input_unavailable",
                    run_id=run.run_id,
                    con_id=row.con_id,
                    symbol=row.symbol,
                    reason="active trading-bar prefix unavailable",
                )
                continue
            session_open = required[0]
            try:
                snapshot = self._cache.get_required_history(
                    instrument, self._FIVE_MINUTES, required, as_of=t0
                )
                if snapshot.status is not HistoryStatus.READY:
                    await self._history.fetch_and_store(
                        instrument,
                        bar_size="5 mins",
                        duration=f"{checkpoint * 5 * 60 + 300} S",
                        what_to_show="TRADES",
                        regular_trading_hours=True,
                        end_time=t0,
                    )
                    snapshot = self._cache.get_required_history(
                        instrument, self._FIVE_MINUTES, required, as_of=t0
                    )
                if snapshot.status is not HistoryStatus.READY:
                    self._logger.debug(
                        "session_hard_input_unavailable",
                        run_id=run.run_id,
                        con_id=instrument.con_id,
                        symbol=instrument.symbol,
                        reason=snapshot.reason,
                    )
                    continue
                features = calculate_session_hard_inputs(
                    snapshot.bars,
                    checkpoint=checkpoint,
                    session_open=session_open,
                    bar_starts=required,
                )
                key = StrategyOpportunityKey(instrument.con_id, row.session, t0)
                assessments[key] = SessionHardAssessment.from_features(
                    checkpoint=checkpoint, features=features
                )
            except (IbkrError, ValueError) as exc:
                self._logger.warning(
                    "session_hard_input_failed",
                    run_id=run.run_id,
                    con_id=instrument.con_id,
                    symbol=instrument.symbol,
                    reason=str(exc),
                )
                continue
        return StrategyContext(
            run_id=run.run_id,
            session_hard=assessments,
            cohort_history=tuple(cohort_history),
        )

    async def bars_for(
        self,
        run: RunConfig,
        instruments: Mapping[int, QualifiedInstrument],
        *,
        session: date,
        now: datetime,
        signals: Sequence[StrategySignal],
    ) -> Mapping[int, Sequence[EntryBar]]:
        causal_now = _aware(now)
        completed_minute = causal_now.replace(second=0, microsecond=0) - timedelta(minutes=1)
        required_by_con_id: dict[int, set[datetime]] = {}
        for signal in signals:
            if (
                signal.run_id != run.run_id
                or signal.session != session
                or signal.underlying_con_id is None
                or signal.status is not SignalStatus.WAITING_FOR_ENTRY
            ):
                continue
            start = _aware(signal.t0)
            if causal_now > start + timedelta(minutes=5):
                self._logger.info(
                    "entry_window_not_replayed",
                    run_id=run.run_id,
                    signal_id=signal.signal_id,
                    con_id=signal.underlying_con_id,
                )
                continue
            end = min(start + timedelta(minutes=4), completed_minute)
            if end < start:
                continue
            required_by_con_id.setdefault(signal.underlying_con_id, set()).update(
                start + timedelta(minutes=index)
                for index in range(int((end - start).total_seconds() // 60) + 1)
            )

        result: dict[int, tuple[EntryBar, ...]] = {}
        for con_id, required_set in required_by_con_id.items():
            instrument = instruments.get(con_id)
            if instrument is None:
                self._logger.warning(
                    "entry_data_unavailable",
                    run_id=run.run_id,
                    con_id=con_id,
                    reason="qualified instrument identity missing",
                )
                continue
            required = tuple(sorted(required_set))
            try:
                snapshot = self._cache.get_required_history(
                    instrument, self._ONE_MINUTE, required, as_of=causal_now
                )
                if snapshot.status is not HistoryStatus.READY:
                    await self._history.fetch_and_store(
                        instrument,
                        bar_size="1 min",
                        duration="900 S",
                        what_to_show="TRADES",
                        regular_trading_hours=True,
                        end_time=causal_now,
                    )
                    snapshot = self._cache.get_required_history(
                        instrument, self._ONE_MINUTE, required, as_of=causal_now
                    )
                if snapshot.status is not HistoryStatus.READY:
                    self._logger.debug(
                        "entry_data_incomplete",
                        run_id=run.run_id,
                        con_id=instrument.con_id,
                        symbol=instrument.symbol,
                        reason=snapshot.reason,
                    )
                result[con_id] = tuple(
                    EntryBar(
                        timestamp=_intraday_timestamp(bar.timestamp),
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                    )
                    for bar in snapshot.bars
                )
            except (IbkrError, ValueError) as exc:
                self._logger.warning(
                    "entry_data_failed",
                    run_id=run.run_id,
                    con_id=instrument.con_id,
                    symbol=instrument.symbol,
                    reason=str(exc),
                )
                continue
        return result


def build_runtime(
    *,
    runs_config_path: str | Path,
    ibkr_config_path: str | Path,
    database_path: str | Path,
    clock: Callable[[], datetime] | None = None,
    logger: Any | None = None,
) -> StockerRuntime:
    """Compose one runtime with explicit sessions for enabled run environments."""

    runs = load_runs_config(runs_config_path)
    configured_environments = tuple(dict.fromkeys(run.environment for run in runs.runs))
    destinations: list[ExecutionDestination] = []
    connections: dict[Environment, IbkrConnection] = {}
    session_identities: set[tuple[str, int, int]] = set()
    for environment in configured_environments:
        try:
            broker_config = load_ibkr_config(ibkr_config_path, environment)
        except ValueError as exc:
            raise ValueError(
                f"configured {environment.value} run requires {environment.value} IBKR config"
            ) from exc
        if broker_config.expected_account is None:
            raise ValueError(
                f"Stage 9 {environment.value} runtime requires {environment.value} expected_account"
            )
        session_identity = (
            broker_config.host,
            broker_config.port,
            broker_config.client_id,
        )
        if session_identity in session_identities:
            raise ValueError("PAPER and LIVE require distinct IBKR session identities")
        session_identities.add(session_identity)
        connection = IbkrConnection(broker_config, execution_enabled=True)
        connections[environment] = connection
        destinations.append(
            ExecutionDestination(environment, broker_config.expected_account, connection)
        )
    execution_router = ExecutionRouter(tuple(destinations))
    data_environment = (
        Environment.PAPER if Environment.PAPER in connections else configured_environments[0]
    )
    market_data_broker = connections[data_environment]
    history_cache = IbkrHistoryCache(database_path)
    prior_context = PriorSessionContextService(
        market_data_broker,
        history_cache,
        PriorSessionContextStore(database_path),
    )
    data_clock = clock or (lambda: datetime.now(tz=UTC))
    current_data = Stage5CurrentDataService(
        market_data_broker,
        history_cache,
        PriorSessionContextExpectedMoveService(prior_context),
        clock=data_clock,
    )
    hv_current_data = Stage5CurrentDataService(
        market_data_broker,
        history_cache,
        IbkrHistoricalVolatilityExpectedMoveService(market_data_broker, clock=data_clock),
        calculation_version=STAGE5_HV_CALCULATION_VERSION,
        clock=data_clock,
    )
    snapshot_store = Stage5SnapshotStore(database_path)
    stage5 = Stage5Analyzer(
        current_data,
        snapshot_store=snapshot_store,
    )
    stage5_by_strategy = {
        SESSION_HARD_METHOD.strategy_version: stage5,
        SESSION_HARD_HV_METHOD.strategy_version: Stage5Analyzer(
            hv_current_data,
            snapshot_store=snapshot_store,
            calculation_version=STAGE5_HV_CALCULATION_VERSION,
        ),
    }
    session_data = IbkrSessionDataSource(market_data_broker, history_cache, logger=logger)
    activity_service = ActivityShortlistService(ActivityShortlistStore(database_path))

    async def qualify(runs_to_prepare: Sequence[RunInstance]) -> Stage5QualificationResult:
        snapshots: dict[str, ActivityShortlistSnapshot] = {}
        now = _aware(data_clock())
        resolver = ExchangeSessionResolver()
        for instance in runs_to_prepare:
            config = instance.config
            if (
                config.screen is None
                or config.screen.method.value != "ACTIVITY_SHORTLIST_V1"
                or config.market_id is None
                or config.cap_bucket is None
            ):
                continue
            market = resolver.resolve(config, now)
            definition = get_market(config.market_id)
            if len(market.active_bar_starts) <= 3:
                snapshots[config.run_id] = ActivityShortlistSnapshot(
                    market_id=config.market_id.value,
                    cap_bucket=config.cap_bucket,
                    cap_bucket_version=config.cap_bucket_version or "CAP_BUCKETS_V1",
                    session=market.session,
                    screen_timestamp=now,
                    profile_id=config.candidate_screen_id or "ACTIVITY_SHORTLIST_V1",
                    profile_version=(config.candidate_screen_version or "ACTIVITY_SHORTLIST_V1"),
                    status=ActivityShortlistStatus.SCANNER_NOT_AVAILABLE,
                    components=(),
                    candidates=(),
                    reason="SCANNER_NOT_AVAILABLE",
                )
                continue
            allowed_symbols = (
                frozenset(item.symbol for item in instance.universe.members)
                if instance.universe.members
                else None
            )
            snapshots[config.run_id] = await activity_service.get_or_create(
                market_data_broker,
                market=definition,
                cap_bucket=CapBucket(config.cap_bucket),
                session=market.session,
                screen_at=market.active_bar_starts[3],
                now=now,
                allowed_symbols=allowed_symbols,
            )
        return await qualify_active_runs(
            market_data_broker,
            runs_to_prepare,
            activity_snapshots=snapshots,
        )

    return StockerRuntime(
        config=runs,
        execution_router=execution_router,
        ledger=ExecutionLedger(database_path),
        store=RuntimeStore(database_path),
        qualify=qualify,
        stage5=stage5,
        stage5_by_strategy=stage5_by_strategy,
        context_provider=session_data,
        entry_source=session_data,
        clock=data_clock,
        logger=logger,
    )


def build_paper_runtime(
    *,
    runs_config_path: str | Path,
    ibkr_config_path: str | Path,
    database_path: str | Path,
    clock: Callable[[], datetime] | None = None,
    logger: Any | None = None,
) -> StockerRuntime:
    """Stage 8 compatibility entry point that can never enable a LIVE run."""

    runs = load_runs_config(runs_config_path)
    if any(run.enabled and run.environment is Environment.LIVE for run in runs.runs):
        raise ValueError(
            "Stage 8 PAPER runtime cannot start enabled LIVE runs; use Stage 9 runtime"
        )
    return build_runtime(
        runs_config_path=runs_config_path,
        ibkr_config_path=ibkr_config_path,
        database_path=database_path,
        clock=clock,
        logger=logger,
    )
