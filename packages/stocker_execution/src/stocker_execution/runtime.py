"""Small Stage 8 application runtime over the completed Stage 1--7 seams."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from stocker_core.config import RunsConfig, load_ibkr_config, load_runs_config
from stocker_core.logging import configure_logging
from stocker_core.runs import Environment, RunConfig, RunInstance, RunManager, RunWindow
from stocker_core.universes import UniverseCatalog
from stocker_data.calendars import get_market_calendar
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import BrokerFill, OrderLifecycle
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
    QualifiedInstrument,
)
from stocker_execution.pre_context import PriorSessionContextService, PriorSessionContextStore
from stocker_execution.session_hard_structure_d import (
    SESSION_HARD_CHECKPOINTS,
    STRATEGY_ID,
    CohortOpportunity,
    EntryBar,
    SessionHardAssessment,
    SessionHardStructureDStrategy,
    SignalStatus,
    StrategyContext,
    StrategyOpportunityKey,
    StrategySignal,
)
from stocker_execution.stage5 import (
    Stage5Analyzer,
    Stage5CurrentDataService,
    Stage5FeatureSnapshot,
    Stage5IneligibleInstrument,
    Stage5QualificationResult,
    Stage5QualifiedRequest,
    Stage5SnapshotStore,
    Stage5Status,
    calculate_session_hard_inputs,
    qualify_active_runs,
)
from stocker_execution.stage7 import (
    ExecutionBroker,
    ExecutionResultCode,
    Stage7ExecutionService,
    Stage7PaperRuntime,
)


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


@dataclass(frozen=True, slots=True)
class MarketSession:
    session: date
    state: MarketSessionState
    opens_at: datetime | None
    closes_at: datetime | None

    def checkpoint_times(self) -> tuple[tuple[int, datetime], ...]:
        if self.opens_at is None or self.closes_at is None:
            return ()
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
class RuntimeStatus:
    application: ApplicationState
    broker_connected: bool
    account: str | None
    runs: tuple[RunStatus, ...]
    counters: RuntimeCounters

    def as_dict(self) -> dict[str, Any]:
        return {
            "application": self.application.value,
            "ibkr_paper": "connected" if self.broker_connected else "disconnected",
            "account": self.account,
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

    def as_text(self) -> str:
        account = self.account or "unavailable"
        lines = [
            f"Application: {self.application.value}",
            f"IBKR PAPER: {'connected' if self.broker_connected else 'disconnected'}",
            f"Account: {account}",
            f"Runs: {sum(run.state is RunRuntimeState.ACTIVE for run in self.runs)} active",
        ]
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

    _DEFAULT = RunWindow(
        start=time(9, 30),
        end=time(16, 0),
        timezone="America/New_York",
        calendar="XNYS",
    )

    def resolve(self, run: RunConfig, now: datetime) -> MarketSession:
        aware_now = _aware(now)
        window = run.session or self._DEFAULT
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
        return MarketSession(session, state, opens_at, closes_at)


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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


class StockerRuntime:
    """One restartable PAPER application with explicit Stage 1--7 composition."""

    _SUPPORTED_STRATEGIES = {"SESSION_HARD", STRATEGY_ID}

    def __init__(
        self,
        *,
        config: RunsConfig,
        broker: RuntimeBroker,
        expected_account: str,
        ledger: ExecutionLedger,
        store: RuntimeStore,
        qualify: Qualifier,
        stage5: Stage5Analyzer,
        context_provider: StrategyContextProvider,
        entry_source: EntryBarSource,
        session_resolver: SessionResolver | None = None,
        clock: Callable[[], datetime] | None = None,
        logger: Any | None = None,
        broker_sync_interval_seconds: float = 5.0,
    ) -> None:
        if not expected_account.strip():
            raise ValueError("Stage 8 PAPER runtime requires expected_account")
        if broker_sync_interval_seconds <= 0.0:
            raise ValueError("broker sync interval must be positive")
        self._config = config
        self._broker = broker
        self._expected_account = expected_account
        self._ledger = ledger
        self._store = store
        self._qualify = qualify
        self._stage5 = stage5
        self._context_provider = context_provider
        self._entry_source = entry_source
        self._session_resolver = session_resolver or ExchangeSessionResolver()
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self._logger = logger or configure_logging()
        self._broker_sync_interval = timedelta(seconds=broker_sync_interval_seconds)
        self._manager = RunManager(UniverseCatalog(config.universes), config.runs)
        self._state = ApplicationState.STOPPED
        self._run_states: dict[str, RunRuntimeState] = {}
        self._run_reasons: dict[str, str] = {}
        self._sessions: dict[str, MarketSession] = {}
        self._execution: dict[str, Stage7ExecutionService] = {}
        self._paper_runtimes: dict[str, Stage7PaperRuntime] = {}
        self._strategies: dict[str, SessionHardStructureDStrategy] = {}
        self._qualification = Stage5QualificationResult((), ())
        self._ready_at: datetime | None = None
        self._last_sync: datetime | None = None
        self._stopping = False

    async def start(self) -> RuntimeStatus:
        """Connect, verify, reconcile, prepare, and only then expose READY."""

        now = _aware(self._clock())
        self._state = ApplicationState.STARTING
        self._stopping = False
        self._execution.clear()
        self._paper_runtimes.clear()
        self._strategies.clear()
        self._qualification = Stage5QualificationResult((), ())
        self._store.recover_interrupted(now)
        self._logger.info("application_start")
        paper_runs: list[RunInstance] = []
        for instance in self._manager.list_runs():
            run = instance.config
            if not run.enabled:
                self._set_run(run.run_id, RunRuntimeState.DISABLED, "disabled by configuration")
            elif run.environment is Environment.LIVE:
                self._set_run(run.run_id, RunRuntimeState.DEGRADED, "LIVE_EXECUTION_DISABLED")
            elif run.strategy not in self._SUPPORTED_STRATEGIES:
                self._set_run(run.run_id, RunRuntimeState.DEGRADED, "unsupported strategy")
            elif run.risk is None:
                self._set_run(run.run_id, RunRuntimeState.DEGRADED, "risk config is required")
            else:
                self._set_run(run.run_id, RunRuntimeState.STARTING, "")
                paper_runs.append(self._manager.start_run(run.run_id))

        if not paper_runs:
            self._state = ApplicationState.DEGRADED
            return self.status()
        for instance in paper_runs:
            self._execution[instance.config.run_id] = Stage7ExecutionService(
                run=instance.config,
                expected_account=self._expected_account,
                broker=self._broker,
                ledger=self._ledger,
                clock=self._clock,
            )
        try:
            session = await self._broker.connect()
        except Exception as exc:
            self._degrade_paper_runs(paper_runs, f"broker unavailable: {exc}")
            self._state = ApplicationState.DEGRADED
            self._logger.error("ibkr_connection_failed", reason=str(exc))
            return self.status()
        self._logger.info(
            "ibkr_connected",
            environment=session.environment.value,
            account=session.masked_account_id,
        )
        if (
            session.environment is not Environment.PAPER
            or session.account_id != self._expected_account
        ):
            self._degrade_paper_runs(paper_runs, "account or environment mismatch")
            self._state = ApplicationState.DEGRADED
            return self.status()

        self._state = ApplicationState.RECONCILING
        all_reconciled = True
        for instance in paper_runs:
            execution = self._execution[instance.config.run_id]
            result = await execution.reconcile()
            self._store.record_reconciliation(
                environment=Environment.PAPER,
                account=self._expected_account,
                connection_epoch=self._broker.connection_epoch,
                ok=result.ok,
                detail=result.detail,
                now=now,
            )
            if not result.ok:
                all_reconciled = False
                self._set_run(instance.config.run_id, RunRuntimeState.DEGRADED, result.detail)
                self._store.increment(instance.config.run_id, now.date(), "reconciliation_issues")
                self._logger.error(
                    "reconciliation_required",
                    run_id=instance.config.run_id,
                    reason=result.detail,
                )
                continue
            strategy = SessionHardStructureDStrategy()
            self._strategies[instance.config.run_id] = strategy
            self._paper_runtimes[instance.config.run_id] = Stage7PaperRuntime(
                strategy=strategy, execution=execution
            )
            market = self._resolve_market(instance, now)
            if market is None:
                continue
            self._sessions[instance.config.run_id] = market
            state = (
                RunRuntimeState.ACTIVE
                if market.state is MarketSessionState.ACTIVE_SESSION
                else RunRuntimeState.READY
            )
            self._set_run(instance.config.run_id, state, "")

        if not all_reconciled:
            self._state = ApplicationState.DEGRADED
            return self.status()
        runnable_runs = tuple(
            instance
            for instance in paper_runs
            if self._run_states.get(instance.config.run_id)
            in {RunRuntimeState.READY, RunRuntimeState.ACTIVE}
        )
        if not runnable_runs:
            self._state = ApplicationState.DEGRADED
            return self.status()
        try:
            self._qualification = await self._qualify(runnable_runs)
        except Exception as exc:
            self._degrade_paper_runs(paper_runs, f"instrument preparation failed: {exc}")
            self._state = ApplicationState.DEGRADED
            return self.status()
        self._ready_at = now
        self._last_sync = now
        self._mark_missed_before(now)
        self._state = ApplicationState.READY
        self._logger.info(
            "application_ready",
            account=session.masked_account_id,
            runs=len(paper_runs),
            qualified=len(self._qualification.requests),
            ineligible=len(self._qualification.ineligible),
        )
        return self.status()

    async def stop(self) -> RuntimeStatus:
        """Stop new decisions, persist local state, and disconnect without cancelling protection."""

        self._stopping = True
        self._state = ApplicationState.STOPPING
        for instance in self._manager.list_runs():
            if self._run_states.get(instance.config.run_id) is not RunRuntimeState.DISABLED:
                self._set_run(instance.config.run_id, RunRuntimeState.STOPPED, "")
                if instance.state.value == "ACTIVE":
                    self._manager.stop_run(instance.config.run_id)
        self._broker.disconnect()
        self._state = ApplicationState.STOPPED
        self._logger.info("application_stop")
        return self.status()

    @property
    def store(self) -> RuntimeStore:
        """Expose the durable operational store for diagnostics and queries."""

        return self._store

    async def poll_once(self) -> RuntimeStatus:
        """Process currently due checkpoints and causal entry observations once."""

        now = _aware(self._clock())
        if self._state is not ApplicationState.READY or self._stopping:
            return self.status()
        if not self._broker.is_connected:
            self._note_disconnect(now)
            return self.status()
        sync_due = self._last_sync is None or now - self._last_sync >= self._broker_sync_interval
        if sync_due and not await self._refresh_execution_state(now):
            return self.status()

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
                missed = (
                    self._ready_at is None
                    or t0 < self._ready_at
                    or now >= t0 + timedelta(minutes=5)
                )
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

    async def reconnect(self) -> RuntimeStatus:
        """Perform one bounded reconnect followed by mandatory reconciliation."""

        now = _aware(self._clock())
        self._state = ApplicationState.RECONCILING
        self._broker.disconnect()
        try:
            session = await self._broker.connect()
        except Exception as exc:
            self._state = ApplicationState.DEGRADED
            for run_id in self._execution:
                self._set_run(run_id, RunRuntimeState.DEGRADED, f"broker unavailable: {exc}")
            return self.status()
        if (
            session.environment is not Environment.PAPER
            or session.account_id != self._expected_account
        ):
            self._state = ApplicationState.DEGRADED
            for run_id in self._execution:
                self._set_run(run_id, RunRuntimeState.DEGRADED, "account or environment mismatch")
            return self.status()

        reconciled = True
        for run_id, execution in self._execution.items():
            result = await execution.refresh_broker_state()
            self._store.record_reconciliation(
                environment=Environment.PAPER,
                account=self._expected_account,
                connection_epoch=self._broker.connection_epoch,
                ok=result.ok,
                detail=result.detail,
                now=now,
            )
            if not result.ok:
                reconciled = False
                self._set_run(run_id, RunRuntimeState.DEGRADED, result.detail)
                self._store.increment(run_id, now.date(), "reconciliation_issues")
            elif run_id not in self._strategies:
                strategy = SessionHardStructureDStrategy()
                self._strategies[run_id] = strategy
                self._paper_runtimes[run_id] = Stage7PaperRuntime(
                    strategy=strategy, execution=execution
                )
        if not reconciled:
            self._state = ApplicationState.DEGRADED
            return self.status()
        active = tuple(
            instance
            for instance in self._manager.list_runs()
            if instance.config.run_id in self._execution
        )
        runnable: list[RunInstance] = []
        for instance in active:
            market = self._resolve_market(instance, now)
            if market is None:
                continue
            self._sessions[instance.config.run_id] = market
            runnable.append(instance)
        try:
            self._qualification = await self._qualify(runnable)
        except Exception as exc:
            self._state = ApplicationState.DEGRADED
            self._degrade_paper_runs(runnable, f"instrument preparation failed: {exc}")
            return self.status()
        self._ready_at = now
        self._last_sync = now
        self._mark_missed_before(now)
        for instance in runnable:
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
        self._state = ApplicationState.READY
        self._logger.info("ibkr_reconnected", account=session.masked_account_id)
        return self.status()

    async def run_forever(self, *, poll_interval_seconds: float = 1.0) -> None:
        """Run the single evaluation loop with one reconnect attempt per interval."""

        if poll_interval_seconds <= 0.0:
            raise ValueError("poll interval must be positive")
        await self.start()
        while not self._stopping:
            if self._state is ApplicationState.READY:
                await self.poll_once()
            elif not self._broker.is_connected and self._execution:
                await self.reconnect()
            await asyncio.sleep(poll_interval_seconds)

    async def market_data_check(self) -> CurrentQuote:
        """Read one qualified current quote for the explicit Stage 8 PAPER smoke path."""

        if self._state is not ApplicationState.READY:
            raise RuntimeError("runtime is not READY")
        request = next(iter(self._qualification.requests), None)
        if request is None:
            raise RuntimeError("no qualified instrument is available for market-data smoke")
        current_quote = getattr(self._broker, "current_quote", None)
        if current_quote is None:
            raise RuntimeError("runtime broker does not expose market data")
        quote: CurrentQuote = await current_quote(request.instrument)
        return quote

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
            execution = next(iter(self._execution.values()), None)
            if execution is not None:
                execution.record_fill(fill)
            self._state = ApplicationState.DEGRADED
            for run_id in self._execution:
                self._set_run(
                    run_id,
                    RunRuntimeState.DEGRADED,
                    "EXECUTION_RECONCILIATION_REQUIRED: unexpected broker fill",
                )
            return False
        execution = self._execution.get(matching.run_id)
        if execution is None or not execution.record_fill(fill):
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
            run_id=matching.run_id,
            order_plan_id=matching.order_plan_id,
            con_id=matching.con_id,
            ibkr_order_id=fill.order_id,
            execution_id=fill.execution_id,
        )
        return True

    def status(self) -> RuntimeStatus:
        now = _aware(self._clock())
        active_records = self._ledger.active_records(Environment.PAPER, self._expected_account)
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
                        record.run_id == run.run_id
                        and record.filled_quantity > record.closed_quantity
                        for record in active_records
                    ),
                )
            )
        return RuntimeStatus(
            application=self._state,
            broker_connected=self._broker.is_connected,
            account=self._broker.account or None,
            runs=tuple(run_statuses),
            counters=self._store.counters(),
        )

    def _set_run(self, run_id: str, state: RunRuntimeState, reason: str) -> None:
        self._run_states[run_id] = state
        self._run_reasons[run_id] = reason

    def _degrade_paper_runs(self, runs: Sequence[RunInstance], reason: str) -> None:
        for instance in runs:
            self._set_run(instance.config.run_id, RunRuntimeState.DEGRADED, reason)

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

    def _mark_missed_before(self, ready_at: datetime) -> None:
        for instance in self._manager.list_runs():
            run_id = instance.config.run_id
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
        now = _aware(self._clock())
        run_ids = {instance.config.run_id for instance in instances}
        requests, ineligible = self._qualification_for(run_ids)
        try:
            rows = await self._stage5.analyze(
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
                strategy.evaluate(valid_rows, context)
            except Exception as exc:
                self._fail_checkpoint(instance, session, t0, str(exc), now)
                continue
            ready_count = sum(row.status is Stage5Status.READY for row in valid_rows)
            self._store.increment(run.run_id, session, "checkpoints_processed")
            self._store.increment(run.run_id, session, "instruments_ready", ready_count)
            self._store.mark_checkpoint(
                run.run_id,
                session,
                t0,
                CheckpointState.COMPLETED,
                f"Stage 5 ready={ready_count}",
                now,
            )
            self._logger.info(
                "checkpoint_processed",
                run_id=run.run_id,
                session=session.isoformat(),
                t0=t0.isoformat(),
                checkpoint=checkpoint,
                instruments_ready=ready_count,
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
            paper_runtime = self._paper_runtimes.get(run.run_id)
            if market is None or paper_runtime is None:
                continue
            requests, _ineligible = self._qualification_for({run.run_id})
            instruments = {request.instrument.con_id: request.instrument for request in requests}
            try:
                strategy = self._strategies[run.run_id]
                bars = await self._entry_source.bars_for(
                    run,
                    instruments,
                    session=market.session,
                    now=now,
                    signals=strategy.signals,
                )
                attempts = await paper_runtime.observe_and_execute(bars, instruments)
            except Exception as exc:
                self._set_run(run.run_id, RunRuntimeState.DEGRADED, str(exc))
                self._logger.error("run_degraded", run_id=run.run_id, reason=str(exc))
                continue
            for attempt in attempts:
                self._store.increment(run.run_id, market.session, "signals")
                if attempt.code is ExecutionResultCode.SUBMITTED:
                    self._store.increment(run.run_id, market.session, "orders")
                    self._logger.info(
                        "order_submitted",
                        run_id=run.run_id,
                        signal_id=attempt.order_plan.signal_id if attempt.order_plan else None,
                        order_plan_id=(
                            attempt.order_plan.order_plan_id if attempt.order_plan else None
                        ),
                        ibkr_order_id=attempt.order_ids.parent if attempt.order_ids else None,
                    )
                elif attempt.code is ExecutionResultCode.BROKER_REJECTED:
                    self._store.increment(run.run_id, market.session, "broker_rejects")
                elif attempt.code not in {
                    ExecutionResultCode.DUPLICATE_ORDER_BLOCKED,
                    ExecutionResultCode.LIVE_EXECUTION_DISABLED,
                }:
                    self._store.increment(run.run_id, market.session, "risk_rejects")
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

    def _note_disconnect(self, now: datetime) -> None:
        self._state = ApplicationState.DEGRADED
        for run_id in self._execution:
            self._set_run(run_id, RunRuntimeState.DEGRADED, "BROKER_DISCONNECTED")
            self._store.increment(run_id, now.date(), "disconnects")
        self._logger.error("ibkr_disconnected")

    async def _refresh_execution_state(self, now: datetime) -> bool:
        """Poll durable broker state without allowing orders during the refresh."""

        self._state = ApplicationState.RECONCILING
        for run_id, execution in self._execution.items():
            result = await execution.refresh_broker_state()
            self._store.record_reconciliation(
                environment=Environment.PAPER,
                account=self._expected_account,
                connection_epoch=self._broker.connection_epoch,
                ok=result.ok,
                detail=result.detail,
                now=now,
            )
            if not result.ok:
                self._set_run(run_id, RunRuntimeState.DEGRADED, result.detail)
                self._store.increment(run_id, now.date(), "reconciliation_issues")
                self._state = ApplicationState.DEGRADED
                return False
        self._last_sync = now
        self._state = ApplicationState.READY
        return True


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

    def __init__(self, ibkr: IbkrConnection, history_cache: IbkrHistoryCache) -> None:
        self._cache = history_cache
        self._history = IbkrHistoryService(ibkr, history_cache)

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
                continue
            t0 = _aware(row.t0)
            session_open = t0 - timedelta(minutes=checkpoint * 5)
            required = tuple(
                session_open + timedelta(minutes=index * 5) for index in range(checkpoint)
            )
            try:
                snapshot = self._cache.get_required_history(
                    instrument, self._FIVE_MINUTES, required, as_of=t0
                )
                if snapshot.status is not HistoryStatus.READY:
                    await self._history.fetch_and_store(
                        instrument,
                        bar_size="5 mins",
                        duration=f"{checkpoint * 5 + 300} S",
                        what_to_show="TRADES",
                        regular_trading_hours=True,
                        end_time=t0,
                    )
                    snapshot = self._cache.get_required_history(
                        instrument, self._FIVE_MINUTES, required, as_of=t0
                    )
                if snapshot.status is not HistoryStatus.READY:
                    continue
                features = calculate_session_hard_inputs(
                    snapshot.bars, checkpoint=checkpoint, session_open=session_open
                )
                key = StrategyOpportunityKey(instrument.con_id, row.session, t0)
                assessments[key] = SessionHardAssessment.from_features(
                    checkpoint=checkpoint, features=features
                )
            except (IbkrError, ValueError):
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
                result[con_id] = tuple(
                    EntryBar(
                        timestamp=_intraday_timestamp(bar.timestamp),
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                    )
                    for bar in snapshot.bars
                )
            except (IbkrError, ValueError):
                continue
        return result


def build_paper_runtime(
    *,
    runs_config_path: str | Path,
    ibkr_config_path: str | Path,
    database_path: str | Path,
    clock: Callable[[], datetime] | None = None,
    logger: Any | None = None,
) -> StockerRuntime:
    """Load configuration and compose the real Stage 2--7 PAPER dependencies once."""

    runs = load_runs_config(runs_config_path)
    broker_config = load_ibkr_config(ibkr_config_path, Environment.PAPER)
    if broker_config.expected_account is None:
        raise ValueError("Stage 8 PAPER runtime requires PAPER expected_account")
    broker = IbkrConnection(broker_config, execution_enabled=True)
    history_cache = IbkrHistoryCache(database_path)
    prior_context = PriorSessionContextService(
        broker,
        history_cache,
        PriorSessionContextStore(database_path),
    )
    data_clock = clock or (lambda: datetime.now(tz=UTC))
    current_data = Stage5CurrentDataService(
        broker,
        history_cache,
        prior_context,
        clock=data_clock,
    )
    stage5 = Stage5Analyzer(
        current_data,
        snapshot_store=Stage5SnapshotStore(database_path),
    )
    session_data = IbkrSessionDataSource(broker, history_cache)

    async def qualify(runs_to_prepare: Sequence[RunInstance]) -> Stage5QualificationResult:
        return await qualify_active_runs(broker, runs_to_prepare)

    return StockerRuntime(
        config=runs,
        broker=broker,
        expected_account=broker_config.expected_account,
        ledger=ExecutionLedger(database_path),
        store=RuntimeStore(database_path),
        qualify=qualify,
        stage5=stage5,
        context_provider=session_data,
        entry_source=session_data,
        clock=data_clock,
        logger=logger,
    )
