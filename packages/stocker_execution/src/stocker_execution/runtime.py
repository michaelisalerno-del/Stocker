"""Small Stage 8 application runtime over the completed Stage 1--7 seams."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

import structlog

from stocker_core.config import IbkrConfig, RunsConfig, load_ibkr_config, load_runs_config
from stocker_core.logging import configure_logging
from stocker_core.markets import (
    LIQUIDITY_ACTIVITY_COMPONENTS,
    MARKET_CATALOGUE,
    MarketId,
    get_market,
)
from stocker_core.methods import content_hash, installed_methods, validate_run_method
from stocker_core.runs import Environment, RunConfig, RunInstance, RunManager
from stocker_core.strategies import SESSION_HARD_HV_METHOD
from stocker_core.universes import UniverseCatalog
from stocker_data.calendars import get_market_calendar
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import BrokerAccountState, BrokerFill, OrderLifecycle
from stocker_execution.history import (
    IbkrHistoryCache,
)
from stocker_execution.ibkr import (
    BrokerSession,
    CurrentQuote,
    IbkrConnection,
    IbkrResourceStatus,
    QualifiedInstrument,
    mask_ibkr_account,
)
from stocker_execution.session_hard_data import IbkrSessionDataSource as IbkrSessionDataSource
from stocker_execution.session_hard_method import SessionHardMethod, TradeEvent
from stocker_execution.session_hard_payoff import (
    CompletedPayoff,
    CostAwareAssessment,
    CostAwareDecision,
    assess_pooled_payoff,
    hypothetical_baseline_outcome,
    is_baseline_payoff_candidate,
    pooled_opportunity_id,
)
from stocker_execution.session_hard_structure_d import (
    SESSION_HARD_CHECKPOINTS,
    CohortOpportunity,
    EntryBar,
    PreMoveBand,
    SignalStatus,
    StrategyContext,
    StrategySignal,
    nominal_exit_prices,
)
from stocker_execution.stage5 import (
    Stage5Analyzer,
    Stage5FeatureSnapshot,
    Stage5IneligibleInstrument,
    Stage5Membership,
    Stage5QualificationResult,
    Stage5QualifiedRequest,
    Stage5SnapshotStore,
    Stage5Status,
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
from stocker_execution.strategy_factory import (
    MethodServices,
    create_method_services,
    create_strategy,
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


HV_EXPECTED_MOVE_PREFETCH_LEAD = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class MarketSession:
    session: date
    state: MarketSessionState
    opens_at: datetime | None
    closes_at: datetime | None
    active_bar_starts: tuple[datetime, ...] = ()

    def checkpoint_times(
        self, checkpoints: Sequence[int] = SESSION_HARD_CHECKPOINTS
    ) -> tuple[tuple[int, datetime], ...]:
        if self.opens_at is None or self.closes_at is None:
            return ()
        if self.active_bar_starts:
            return tuple(
                (checkpoint, self.active_bar_starts[checkpoint])
                for checkpoint in checkpoints
                if checkpoint < len(self.active_bar_starts)
            )
        return tuple(
            (checkpoint, self.opens_at + timedelta(minutes=checkpoint * 5))
            for checkpoint in checkpoints
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
    next_checkpoint: datetime | None = None
    last_scheduled_checkpoint: datetime | None = None
    evaluation_checkpoint: datetime | None = None
    evaluation_completed: int = 0
    evaluation_total: int = 0
    evaluation_state: str = "IDLE"
    preparing_history: bool = False
    trade_stream_unavailable: int = 0


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
    trade_errors: dict[int, str]

    def prepare_trades(self, instrument: QualifiedInstrument) -> None: ...

    def release_trades(self, con_id: int) -> None: ...

    def release_unused_trades(self, retained: set[int]) -> None: ...

    async def trades_for(
        self, instruments: Mapping[int, QualifiedInstrument], signals: Sequence[StrategySignal]
    ) -> Mapping[int, Sequence[TradeEvent]]: ...

    async def cohort_bars(
        self, instrument: QualifiedInstrument, signal: StrategySignal, now: datetime
    ) -> Sequence[EntryBar]: ...

    async def bars_for(
        self,
        run: RunConfig,
        instruments: Mapping[int, QualifiedInstrument],
        *,
        session: date,
        now: datetime,
        signals: Sequence[StrategySignal],
        fetch_missing: bool = True,
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
                CREATE TABLE IF NOT EXISTS method_runs (
                    run_id TEXT PRIMARY KEY,
                    configuration TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS method_cohort_labels (
                    signal_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS method_run_configurations (
                    run_id TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    configuration TEXT NOT NULL
                );
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
                CREATE TABLE IF NOT EXISTS runtime_session_hard_payoffs (
                    signal_id TEXT PRIMARY KEY,
                    opportunity_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    con_id INTEGER NOT NULL,
                    signal_timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    assessment TEXT,
                    completion_timestamp TEXT,
                    gross_r REAL,
                    actually_executed INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS runtime_payoff_completion
                    ON runtime_session_hard_payoffs(completion_timestamp);
                CREATE TABLE IF NOT EXISTS runtime_signals (
                    signal_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    session TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS runtime_signals_checkpoint
                    ON runtime_signals(run_id, session, json_extract(payload, '$.t0'));
                """
            )

    def save_method_run(self, run: RunConfig, now: datetime) -> None:
        payload = run.model_dump_json()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT configuration FROM method_runs WHERE run_id = ?", (run.run_id,)
            ).fetchone()
            if existing is not None:
                saved = json.loads(existing[0])
                if saved.get("method_spec_hash") != run.method_spec_hash:
                    raise ValueError("Existing run belongs to a different method specification")
            if existing is None or json.loads(existing[0]) != json.loads(payload):
                connection.execute(
                    "INSERT INTO method_run_configurations VALUES (?, ?, ?)",
                    (run.run_id, now.isoformat(), payload),
                )
            connection.execute(
                "INSERT OR IGNORE INTO method_runs VALUES (?, ?, ?, ?, ?, ?)",
                (run.run_id, payload, now.isoformat(), now.isoformat(), "CONFIGURED", ""),
            )
            connection.execute(
                "UPDATE method_runs SET configuration = ? WHERE run_id = ?",
                (payload, run.run_id),
            )

    def save_cohort_labels(self, signals: Sequence[StrategySignal]) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO method_cohort_labels VALUES (?, ?, ?)",
                [(s.signal_id, s.run_id, json.dumps(_signal_payload(s))) for s in signals],
            )

    def load_cohort_labels(self, run_id: str) -> tuple[StrategySignal, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM method_cohort_labels WHERE run_id = ?", (run_id,)
            ).fetchall()
        return tuple(_signal_from_payload(json.loads(row[0])) for row in rows)

    def method_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM method_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["configuration"] = json.loads(result["configuration"])
        config = result["configuration"]
        result.update(
            market=config["market_id"],
            method_id=config["strategy_id"],
            method_version=config["strategy_version"],
            method_spec_hash=config["method_spec_hash"],
            universe_snapshot_sha256=content_hash(config["universe_snapshot"]),
            data_source="IBKR",
            broker_mode=config["environment"],
        )
        signals = self.load_signals(run_id)
        result["candidate_count"] = len(signals)
        result["screened"] = len(signals)
        result["universe_count"] = len((config.get("universe_snapshot") or {}).get("members", []))
        result["qualified"] = sum(s.session_hard_qualified for s in signals)
        result["vetoed"] = sum(
            s.q1_eligible is False or s.reason == "COHORT_MID_VETO" for s in signals
        )
        result["armed"] = sum(s.status is SignalStatus.WAITING_FOR_ENTRY for s in signals)
        result["triggered"] = sum(s.status is SignalStatus.ENTRY_TRIGGERED for s in signals)
        result["sessions"] = sorted({s.session.isoformat() for s in signals})
        result["errors"] = [s.reason for s in signals if "UNAVAILABLE" in s.reason]
        with self._connect() as connection:
            result["configuration_history"] = [
                {"saved_at": r[0], "configuration": json.loads(r[1])}
                for r in connection.execute(
                    "SELECT saved_at, configuration FROM method_run_configurations "
                    "WHERE run_id = ? ORDER BY saved_at",
                    (run_id,),
                )
            ]
            result["errors"].extend(
                r[0]
                for r in connection.execute(
                    "SELECT detail FROM runtime_checkpoints WHERE run_id = ? AND state = 'FAILED'",
                    (run_id,),
                )
            )
            has_ledger = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'execution_plans'"
            ).fetchone()
            positions = (
                connection.execute(
                    "SELECT order_plan_id, filled_quantity, closed_quantity, status "
                    "FROM execution_plans WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
                if has_ledger
                else []
            )
        result["active_positions"] = [r[0] for r in positions if r[1] > r[2]]
        result["completed_positions"] = [r[0] for r in positions if r[3] == "CLOSED"]
        return result

    def set_method_run_state(self, run_id: str, state: str, reason: str, now: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE method_runs SET updated_at = ?, status = ?, reason = ? WHERE run_id = ?",
                (now.isoformat(), state, reason, run_id),
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

    def load_signals(
        self, run_id: str, *, session: date | None = None, t0: datetime | None = None,
        signal_ids: Sequence[str] | None = None,
        con_ids: Sequence[int] | None = None, status: SignalStatus | None = None,
    ) -> tuple[StrategySignal, ...]:
        """Load signals for one run; callers apply the live session/window rules."""

        conditions = ["run_id = ?"]
        params: list[Any] = [run_id]
        if session is not None:
            conditions.append("session = ?")
            params.append(session.isoformat())
        if t0 is not None:
            conditions.append("json_extract(payload, '$.t0') = ?")
            params.append(_aware(t0).isoformat())
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)
        if con_ids is not None:
            if not con_ids:
                return ()
            conditions.append("json_extract(payload, '$.underlying_con_id') IN ("
                              + ",".join("?" for _ in con_ids) + ")")
            params.extend(con_ids)
        if signal_ids is not None:
            if not signal_ids:
                return ()
            conditions.append("signal_id IN (" + ",".join("?" for _ in signal_ids) + ")")
            params.extend(signal_ids)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM runtime_signals WHERE " + " AND ".join(conditions)
                + " ORDER BY session, signal_id",
                params,
            ).fetchall()
        return tuple(_signal_from_payload(json.loads(str(row["payload"]))) for row in rows)

    def signal_counts(self, run_id: str, session: date, t0: datetime | None) -> dict[str, int]:
        """Read checkpoint totals without materializing candidate objects or old sessions."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT count(*) AS screened,
                    coalesce(sum(status != 'NOT_QUALIFIED'), 0) AS qualified,
                    coalesce(sum(json_extract(payload, '$.q1_eligible') = 0
                        OR json_extract(payload, '$.reason') = 'COHORT_MID_VETO'), 0) AS vetoed,
                    coalesce(sum(status = 'WAITING_FOR_ENTRY'), 0) AS armed,
                    coalesce(sum(status = 'ENTRY_TRIGGERED'), 0) AS triggered
                FROM runtime_signals WHERE run_id = ? AND session = ?
                    AND json_extract(payload, '$.t0') = ?""",
                (run_id, session.isoformat(), _aware(t0).isoformat() if t0 else None),
            ).fetchone()
        return {key: int(value) for key, value in dict(row).items()}

    def register_payoff(
        self,
        signal: StrategySignal,
        instrument: QualifiedInstrument,
        run: RunConfig,
        cost_bps: float | None,
    ) -> None:
        """Record every filled baseline candidate before admission or risk selection."""
        if not is_baseline_payoff_candidate(signal):
            raise ValueError("only exact baseline-fillable Session HARD candidates enter the pool")
        if signal.underlying_con_id != instrument.con_id:
            raise ValueError("baseline instrument identity mismatch")
        if signal.signal_timestamp is None:
            raise ValueError("baseline signal timestamp is required")
        stop, _target = nominal_exit_prices(signal)
        payload = {
            "signal": _signal_payload(signal),
            "instrument": asdict(instrument),
            "run": run.model_dump(mode="json"),
            "baseline_eligible": True,
            "entry_reference_price": signal.entry_reference,
            "initial_stop_price": stop,
            "estimated_round_trip_cost_bps": cost_bps,
            "admission_decision": "TRADE_BASELINE_PRE_HURDLE" if cost_bps is None else None,
        }
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO runtime_session_hard_payoffs
                (signal_id, opportunity_id, symbol, con_id, signal_timestamp, payload)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    signal.signal_id,
                    pooled_opportunity_id(signal),
                    signal.symbol,
                    signal.underlying_con_id,
                    _aware(signal.signal_timestamp).isoformat(),
                    json.dumps(payload, sort_keys=True),
                ),
            )

    def backfill_payoffs(self, runs: Sequence[RunConfig], history: IbkrHistoryCache) -> None:
        """Register saved exact HV entries; reconstruct gross outcomes from IBKR bars only.

        These signals predate admission. Their historical cost/estimate is unknown
        and remains null, rather than being reconstructed using today's assumption.
        """
        for run in runs:
            for signal in self.load_signals(run.run_id):
                if not is_baseline_payoff_candidate(signal) or signal.underlying_con_id is None:
                    continue
                instrument = history.qualified_instrument(signal.underlying_con_id)
                if instrument is not None:
                    self.register_payoff(signal, instrument, run, cost_bps=None)

    def pending_payoffs(
        self,
    ) -> tuple[tuple[StrategySignal, QualifiedInstrument, RunConfig], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT signal_id, payload FROM runtime_session_hard_payoffs
                WHERE completion_timestamp IS NULL ORDER BY signal_timestamp, signal_id"""
            ).fetchall()
        pending = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
                signal = replace(_signal_from_payload(payload["signal"]), baseline_eligible=True)
                pending.append(
                    (
                        signal,
                        QualifiedInstrument(**payload["instrument"]),
                        RunConfig.model_validate(payload["run"]),
                    )
                )
            except (ValueError, TypeError, KeyError) as exc:
                structlog.get_logger(__name__).warning(
                    "session_hard_shadow_invalid", signal_id=row["signal_id"], reason=str(exc)
                )
        return tuple(pending)

    def complete_payoff(self, signal_id: str, outcome: CompletedPayoff) -> None:
        """Persist immutable gross outcomes; costs and broker P&L never enter the pool."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runtime_session_hard_payoffs WHERE signal_id = ?", (signal_id,)
            ).fetchone()
            if row is None or row["opportunity_id"] != outcome.opportunity_id:
                raise ValueError("completed payoff has no matching baseline opportunity")
            # A touch-bar outcome may complete exactly when its entry becomes
            # actionable, but is still excluded from that admission's history.
            if outcome.completion_timestamp < datetime.fromisoformat(row["signal_timestamp"]):
                raise ValueError("baseline outcome completed before its signal")
            if row["completion_timestamp"] is not None:
                if (
                    row["completion_timestamp"] != _aware(outcome.completion_timestamp).isoformat()
                    or row["gross_r"] != outcome.gross_r
                ):
                    raise ValueError("completed baseline outcomes are immutable")
                return
            connection.execute(
                """UPDATE runtime_session_hard_payoffs
                SET completion_timestamp = ?, gross_r = ?
                WHERE signal_id = ? AND completion_timestamp IS NULL""",
                (_aware(outcome.completion_timestamp).isoformat(), outcome.gross_r, signal_id),
            )

    def assess_payoff(self, signal_id: str) -> CostAwareAssessment:
        """Atomically freeze the original decision from strictly prior pooled completions."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runtime_session_hard_payoffs WHERE signal_id = ?", (signal_id,)
            ).fetchone()
            if row is None:
                raise ValueError("payoff admission requires a registered baseline opportunity")
            if row["assessment"] is not None:
                values = json.loads(row["assessment"])
                values["decision"] = CostAwareDecision(values["decision"])
                return CostAwareAssessment(**values)
            payload = json.loads(row["payload"])
            prior = connection.execute(
                """SELECT opportunity_id, completion_timestamp, gross_r
                FROM runtime_session_hard_payoffs
                WHERE completion_timestamp < ? ORDER BY completion_timestamp, opportunity_id""",
                (row["signal_timestamp"],),
            ).fetchall()
            assessment = assess_pooled_payoff(
                opportunity_id=row["opportunity_id"],
                signal_timestamp=datetime.fromisoformat(row["signal_timestamp"]),
                entry_reference_price=payload["entry_reference_price"],
                initial_stop_price=payload["initial_stop_price"],
                estimated_round_trip_cost_bps=payload["estimated_round_trip_cost_bps"],
                observations=(
                    CompletedPayoff(
                        item["opportunity_id"],
                        datetime.fromisoformat(item["completion_timestamp"]),
                        item["gross_r"],
                    )
                    for item in prior
                ),
            )
            connection.execute(
                """UPDATE runtime_session_hard_payoffs SET assessment = ?
                WHERE signal_id = ?""",
                (json.dumps(asdict(assessment), sort_keys=True), signal_id),
            )
        return assessment

    def payoff_audit(self) -> tuple[dict[str, Any], ...]:
        """Expose admission, hypothetical completion, and actual fills separately."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runtime_session_hard_payoffs ORDER BY signal_timestamp, signal_id"
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def record_payoff_executions(self, signal_ids: Sequence[str]) -> None:
        if not signal_ids:
            return
        with self._connect() as connection:
            connection.executemany(
                """UPDATE runtime_session_hard_payoffs SET actually_executed = 1
                WHERE signal_id = ? AND actually_executed = 0""",
                ((signal_id,) for signal_id in signal_ids),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


class StockerRuntime:
    """One restartable application with per-run execution routing."""

    def __init__(
        self,
        *,
        config: RunsConfig,
        ledger: ExecutionLedger,
        store: RuntimeStore,
        qualify: Qualifier,
        stage5: Stage5Analyzer,
        stage5_by_strategy: Mapping[str, Stage5Analyzer] | None = None,
        method_services: Mapping[str, MethodServices] | None = None,
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
            SESSION_HARD_HV_METHOD.strategy_version: stage5,
            **dict(stage5_by_strategy or {}),
        }
        self._context_provider = context_provider
        self._entry_source = entry_source
        self._method_services = dict(method_services or {})
        self._default_method_services = MethodServices(
            stage5, context_provider, entry_source, lambda market: market.checkpoint_times()
        )
        self._payoff_history_task: asyncio.Task[None] | None = None
        self._payoff_fetch_minute: datetime | None = None
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
        self._strategies: dict[str, SessionHardMethod] = {}
        self._qualification = Stage5QualificationResult((), ())
        self._expected_move_prepared: set[tuple[str, date, datetime, tuple[int, ...]]] = set()
        self._session_history_prepared: dict[str, tuple[date, tuple[int, ...]]] = {}
        self._checkpoint_tasks: dict[tuple[str, date, datetime], asyncio.Task[None]] = {}
        self._checkpoint_task_owners: dict[tuple[str, date, datetime], frozenset[str]] = {}
        self._checkpoint_progress: dict[str, dict[str, Any]] = {}
        self._trade_stream_failures: dict[str, int] = {}
        self._expected_move_tasks: dict[
            tuple[str, date, tuple[int, ...]], tuple[asyncio.Task[None], frozenset[str]]
        ] = {}
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
        self._session_history_prepared.clear()
        self._expected_move_tasks.clear()
        for environment in self._environment_ready:
            self._environment_ready[environment] = False
            self._environment_reconciled[environment] = False
        self._store.recover_interrupted(now)
        self._logger.info("application_start")
        configured_runs: list[RunInstance] = []
        for instance in self._manager.list_runs():
            run = instance.config
            method_error = ""
            if run.enabled:
                try:
                    validate_run_method(run)
                except (ValueError, OSError) as exc:
                    method_error = str(exc)
            if run.method_spec is not None:
                self._store.save_method_run(run, now)
            if not run.enabled:
                self._set_run(run.run_id, RunRuntimeState.DISABLED, "disabled by configuration")
            elif method_error:
                self._set_run(run.run_id, RunRuntimeState.DEGRADED, method_error)
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
        checkpoints = list(self._checkpoint_tasks.values())
        for task in checkpoints:
            task.cancel()
        await asyncio.gather(*checkpoints, return_exceptions=True)
        self._checkpoint_tasks.clear()
        self._checkpoint_task_owners.clear()
        preparation = [task for task, _owners in self._expected_move_tasks.values()]
        for task in preparation:
            task.cancel()
        await asyncio.gather(*preparation, return_exceptions=True)
        self._expected_move_tasks.clear()
        if self._payoff_history_task is not None:
            self._payoff_history_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._payoff_history_task
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
        for run in config.runs:
            if run.run_id in changed_run_ids and run.enabled:
                validate_run_method(run)
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
            for key, task in self._checkpoint_tasks.items():
                if not any(updated_by_id[owner].enabled
                           for owner in self._checkpoint_task_owners[key]):
                    task.cancel()
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
                if updated.method_spec is not None:
                    self._store.save_method_run(updated, now)
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
                        current.universe_source != updated.universe_source,
                        current.discovery_profile != updated.discovery_profile,
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
                result = await self._reconcile_execution_services(
                    tuple(
                        (identity, service)
                        for identity, service in self._execution_services()
                        if service.run_environment is updated.environment
                    )
                )
                self._environment_reconciled[updated.environment] = result.ok
                self._environment_ready[updated.environment] = result.ok
                self._state = (
                    ApplicationState.READY
                    if any(self._environment_ready.values())
                    else ApplicationState.DEGRADED
                )
                self._store.record_reconciliation(
                    environment=updated.environment,
                    account=destination.expected_account,
                    connection_epoch=destination.broker.connection_epoch,
                    ok=result.ok,
                    detail=result.detail,
                    now=_aware(self._clock()),
                )
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
            with self._timing("cycle"):
                return await self._poll_once()

    @contextmanager
    def _timing(self, operation: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            if elapsed_ms >= 1000:
                self._logger.info(
                    "runtime_stage_timing",
                    operation=operation,
                    elapsed_ms=round(elapsed_ms, 1),
                )

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
        if sync_due:
            with self._timing("broker_reconciliation"):
                if not await self._refresh_execution_state(now):
                    return self.status()

        # Existing waiting entries precede bulk work. Reconciliation stays first.
        with self._timing("entries_before_preparation"):
            await self._observe_entries(_aware(self._clock()))
        with self._timing("session_and_shortlist_refresh"):
            await self._refresh_activity_sessions(_aware(self._clock()))
            await self._refresh_scheduled_activity_shortlists(_aware(self._clock()))
        with self._timing("expected_move_preparation"):
            await self._prepare_upcoming_expected_moves(_aware(self._clock()))

        for key, task in tuple(self._checkpoint_tasks.items()):
            if task.done():
                if not task.cancelled() and (error := task.exception()) is not None:
                    self._logger.error("checkpoint_task_failed", reason=str(error))
                del self._checkpoint_tasks[key]
                self._checkpoint_task_owners.pop(key, None)
            elif _aware(self._clock()) >= key[2] + timedelta(minutes=5):
                task.cancel()
        now = _aware(self._clock())
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
            for checkpoint, t0 in self._services_for(run).checkpoints(market):
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
            with self._timing("checkpoint_evaluation"):
                await self._evaluate_group(instances, session=session, t0=t0, checkpoint=checkpoint)
        with self._timing("entries_after_checkpoints"):
            await self._observe_entries(_aware(self._clock()))
        return self.status()

    def _services_for(self, run: RunConfig) -> MethodServices:
        if not self._method_services:
            return self._default_method_services
        return self._method_services[str(run.strategy_version)]

    async def _prepare_upcoming_expected_moves(self, now: datetime) -> None:
        enabled = {i.config.run_id for i in self._manager.list_runs() if i.config.enabled}
        self._session_history_prepared = {
            run_id: key for run_id, key in self._session_history_prepared.items()
            if run_id in enabled
        }
        for pending_key, (task, owners) in tuple(self._expected_move_tasks.items()):
            if task.done():
                if not task.cancelled() and (error := task.exception()) is not None:
                    self._logger.warning("expected_move_preparation_failed", reason=str(error))
                del self._expected_move_tasks[pending_key]
            elif not owners.intersection(enabled):
                task.cancel()
        self._expected_move_prepared = {
            key for key in self._expected_move_prepared if key[2] >= now
        }
        upcoming: dict[tuple[str, date, datetime], set[str]] = {}
        for instance in self._manager.list_runs():
            run_id = instance.config.run_id
            if self._run_states.get(run_id) not in {
                RunRuntimeState.READY,
                RunRuntimeState.ACTIVE,
            }:
                continue
            strategy = self._strategies.get(run_id)
            if strategy is None:
                continue
            lead = (
                (instance.config.method_spec or {})
                .get("data_requirements", {})
                .get("prefetch_lead_minutes", 0)
            )
            market = self._sessions.get(run_id) or self._resolve_market(instance, now)
            if market is None:
                continue
            services = self._services_for(instance.config)
            schedule = services.checkpoints(market)
            if services.prepare_history_on_ready and schedule:
                requests, _ = self._qualification_for({run_id})
                history_key = market.session, tuple(sorted(r.instrument.con_id for r in requests))
                if self._session_history_prepared.get(run_id) != history_key:
                    self._session_history_prepared[run_id] = history_key
                    self._start_expected_move_preparation(
                        strategy.strategy_version, market.session, schedule[0][1],
                        requests, {run_id},
                    )
            for _checkpoint, t0 in schedule:
                if not now < t0 <= now + timedelta(minutes=lead):
                    continue
                if self._store.checkpoint_state(run_id, market.session, t0) is not None:
                    continue
                upcoming.setdefault((strategy.strategy_version, market.session, t0), set()).add(
                    run_id
                )

        for (version, session, t0), run_ids in sorted(upcoming.items()):
            source = self._services_for(self._manager.get_run(sorted(run_ids)[0]).config).entries
            requests, _ineligible = self._qualification_for(run_ids)
            con_ids = tuple(sorted(request.instrument.con_id for request in requests))
            key = (version, session, t0, con_ids)
            if key in self._expected_move_prepared:
                continue
            self._expected_move_prepared.add(key)
            failures = 0
            if hasattr(source, "prepare_trades"):
                for request in requests:
                    try:
                        source.prepare_trades(request.instrument)
                    except Exception as exc:
                        failures += 1
                        self._logger.debug(
                            "trade_stream_unavailable",
                            symbol=request.instrument.symbol,
                            reason=str(exc),
                        )
            for run_id in run_ids:
                self._trade_stream_failures[run_id] = failures
            if failures:
                self._logger.warning("checkpoint_stream_capacity", unavailable=failures,
                                     requested=len(requests), t0=t0.isoformat())
            self._start_expected_move_preparation(version, session, t0, requests, run_ids)

    def _start_expected_move_preparation(
        self, version: str, session: date, t0: datetime,
        requests: Sequence[Stage5QualifiedRequest], run_ids: set[str],
    ) -> None:
        # Share bounded background history work; this never opens live trade streams.
        task_key = version, session, tuple(sorted(r.instrument.con_id for r in requests))
        if task_key not in self._expected_move_tasks:
            self._expected_move_tasks[task_key] = (
                asyncio.create_task(
                    self._stage5_by_strategy[version].prepare_expected_moves(
                        requests, session=session, t0=t0
                    )
                ), frozenset(run_ids),
            )
        else:
            task, owners = self._expected_move_tasks[task_key]
            self._expected_move_tasks[task_key] = (task, owners | frozenset(run_ids))

    async def _refresh_scheduled_activity_shortlists(self, now: datetime) -> None:
        """Qualify fixed-time activity screens once they become causally due."""

        pending_run_ids = {
            membership.run_id
            for item in self._qualification.ineligible
            if item.symbol in {r.activity_profile_id for r in self._config.runs}
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
                or not run.uses_activity_shortlist
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
            market = self._sessions.get(instance.config.run_id)
            if (
                instance.config.uses_activity_shortlist
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
            if failure.symbol not in {
                "HOT_BY_VOLUME", "ACTIVITY_SHORTLIST_V1", "ACTIVITY_CAPACITY_V2",
                "ACTIVITY_LIQUIDITY_V1", "ACTIVITY_LIQUIDITY_V2",
            }
            and failure.symbol not in {r.activity_profile_id for r in self._config.runs}
            for membership in failure.memberships
            if membership.run_id in run_markets
        }
        from stocker_core.methods import installed_methods
        from stocker_execution.discovery import scanner_requests

        discovery_readiness = {}
        for method in installed_methods():
            for market_id, profile in method.discovery_profiles:
                try:
                    scanner_requests(profile, get_market(market_id), capabilities)
                    discovery_readiness[market_id] = "AVAILABLE"
                except ValueError as exc:
                    discovery_readiness[market_id] = str(exc)
        return {
            market.market_id.value: (
                discovery_readiness[market.market_id]
                if market.market_id in discovery_readiness else
                "CONTRACT_QUALIFICATION_FAILED"
                if market.market_id in failed_markets and market.market_id not in qualified_markets
                else "AVAILABLE"
                if market.scanner_location in capabilities.locations
                and capabilities.supports_instrument(
                    market.scanner_location, market.scanner_instrument
                )
                and len(
                    {component.value for component in LIQUIDITY_ACTIVITY_COMPONENTS}
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

        matching = self._ledger.record_for_order(fill.environment, fill.account, fill.order_id)
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
        accepted = (
            matching_execution.record_fill(fill) if matching_execution is not None
            else self._ledger.record_fill(fill)
        )
        if not accepted:
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
            schedule = self._services_for(run).checkpoints(market) if market else ()
            progress = self._checkpoint_progress.get(run.run_id, {})
            if progress and market and progress["t0"].date() != market.session:
                progress = {}
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
                    next_checkpoint=next((t0 for _, t0 in schedule if t0 > now), None),
                    last_scheduled_checkpoint=schedule[-1][1] if schedule else None,
                    evaluation_checkpoint=progress.get("t0"),
                    evaluation_completed=progress.get("completed", 0),
                    evaluation_total=progress.get("total", 0),
                    evaluation_state=progress.get("state", "IDLE"),
                    preparing_history=any(
                        run.run_id in owners and not task.done()
                        for task, owners in self._expected_move_tasks.values()
                    ),
                    trade_stream_unavailable=self._trade_stream_failures.get(run.run_id, 0),
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
        self._store.set_method_run_state(run_id, state.value, reason, _aware(self._clock()))
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
        strategy_id = str(run.strategy_id or SESSION_HARD_HV_METHOD.strategy_id)
        strategy_version = str(run.strategy_version or SESSION_HARD_HV_METHOD.strategy_version)
        strategy = create_strategy(
            strategy_id, strategy_version, run.market_id or MarketId.US_ALL, clock=self._clock
        )
        strategy.restore_runtime_state(self._store, run_id)
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
            for checkpoint, t0 in self._services_for(instance.config).checkpoints(market):
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
            if self._services_for(grouped[version][0].config).incremental_checkpoints:
                key = version, session, t0
                self._checkpoint_task_owners[key] = frozenset(
                    instance.config.run_id for instance in grouped[version]
                )
                self._checkpoint_tasks[key] = asyncio.create_task(
                    self._evaluate_incrementally(grouped[version], stage5=stage5,
                                                 session=session, t0=t0, checkpoint=checkpoint)
                )
                continue
            await self._evaluate_strategy_group(
                grouped[version],
                stage5=stage5,
                session=session,
                t0=t0,
                checkpoint=checkpoint,
            )

    async def _evaluate_incrementally(
        self, instances: Sequence[RunInstance], *, stage5: Stage5Analyzer,
        session: date, t0: datetime, checkpoint: int,
    ) -> None:
        """Publish bounded batches without holding the scheduler during broker I/O."""
        run_ids = {item.config.run_id for item in instances}
        requests, ineligible = self._qualification_for(run_ids)
        for instance in instances:
            run_id = instance.config.run_id
            self._checkpoint_progress[run_id] = {
                "t0": t0, "completed": 0, "total": sum(
                    any(m.run_id == run_id for m in request.memberships) for request in requests
                ), "state": "EVALUATING",
            }
        outcome = CheckpointState.COMPLETED
        reason = ""
        try:
            # The method's existing half-open entry window also bounds data work.
            remaining = (t0 + timedelta(minutes=5) - _aware(self._clock())).total_seconds()
            async with asyncio.timeout(max(0, remaining)):
                for offset in range(0, max(1, len(requests)), 4):
                    active = tuple(i for i in instances if self._checkpoint_owner_active(i))
                    if not active:
                        raise asyncio.CancelledError
                    if _aware(self._clock()) >= t0 + timedelta(minutes=5):
                        raise TimeoutError
                    await self._evaluate_strategy_group(
                        active, stage5=stage5, session=session, t0=t0, checkpoint=checkpoint,
                        batch=requests[offset:offset + 4],
                        batch_ineligible=ineligible if offset == 0 else (),
                        background=True,
                    )
                    await asyncio.sleep(0)
        except (TimeoutError, asyncio.CancelledError):
            outcome = CheckpointState.FAILED
            reason = "ENTRY_WINDOW_ELAPSED_OR_RUN_STOPPED: unfinished inputs were not evaluated"
        except Exception as exc:
            outcome = CheckpointState.FAILED
            reason = str(exc)
        finally:
            for instance in instances:
                run_id = instance.config.run_id
                progress = self._checkpoint_progress[run_id]
                progress["state"] = (
                    "COMPLETED" if outcome is CheckpointState.COMPLETED else "INCOMPLETE"
                )
                self._store.mark_checkpoint(
                    run_id, session, t0, outcome, reason, _aware(self._clock())
                )
                if outcome is CheckpointState.COMPLETED:
                    self._store.increment(run_id, session, "checkpoints_processed")
                self._logger.info("checkpoint_finished", run_id=run_id, t0=t0.isoformat(),
                                  completed=progress["completed"], total=progress["total"],
                                  state=progress["state"], reason=reason)

    def _checkpoint_owner_active(self, instance: RunInstance) -> bool:
        run = instance.config
        current = self._manager.get_run(run.run_id).config
        return (
            not self._stopping and current.enabled and current == run
            and self._environment_ready.get(run.environment, False)
            and self._run_states.get(run.run_id) in {RunRuntimeState.READY, RunRuntimeState.ACTIVE}
        )

    async def _evaluate_strategy_group(
        self, instances: Sequence[RunInstance], *, stage5: Stage5Analyzer,
        session: date, t0: datetime, checkpoint: int,
        batch: Sequence[Stage5QualifiedRequest] | None = None,
        batch_ineligible: Sequence[Stage5IneligibleInstrument] = (),
        background: bool = False,
    ) -> None:
        now = _aware(self._clock())
        run_ids = {instance.config.run_id for instance in instances}
        requests, ineligible = self._qualification_for(run_ids)
        if batch is not None:
            requests, ineligible = tuple(batch), tuple(batch_ineligible)
        try:
            rows = await stage5.analyze(requests, ineligible=ineligible, session=session, t0=t0)
        except Exception as exc:
            if background:
                raise
            for instance in instances:
                self._fail_checkpoint(instance, session, t0, str(exc), now)
            return
        for instance in instances:
            run = instance.config
            valid_rows = tuple(row for row in rows if run.run_id in row.run_ids)
            if any(row.session != session or row.t0 != t0 or row.t0.tzinfo is None
                   for row in valid_rows):
                if background:
                    raise ValueError("STALE_OR_SESSION_MISMATCHED_INPUT")
                self._fail_checkpoint(
                    instance, session, t0, "STALE_OR_SESSION_MISMATCHED_INPUT", now
                )
                continue
            try:
                context = await self._services_for(run).context.context_for(
                    run, valid_rows, checkpoint,
                    {request.instrument.con_id: request.instrument for request in requests},
                    self._store.cohort_history(run.run_id),
                )
                if context.run_id != run.run_id:
                    raise ValueError("strategy context run identity mismatch")
            except Exception as exc:
                if background:
                    raise
                self._fail_checkpoint(instance, session, t0, str(exc), now)
                continue
            if background:
                async with self._cycle_lock:
                    if not self._checkpoint_owner_active(instance):
                        continue
                    if _aware(self._clock()) >= t0 + timedelta(minutes=5):
                        raise TimeoutError
                    self._record_checkpoint_rows(run, valid_rows, context, session, t0, checkpoint,
                                                 finalize=False)
                    self._checkpoint_progress[run.run_id]["completed"] += sum(
                        row.con_id is not None for row in valid_rows
                    )
            else:
                self._record_checkpoint_rows(run, valid_rows, context, session, t0, checkpoint)

    def _record_checkpoint_rows(
        self, run: RunConfig, valid_rows: Sequence[Stage5FeatureSnapshot],
        context: StrategyContext, session: date, t0: datetime, checkpoint: int,
        *, finalize: bool = True,
    ) -> None:
        now = _aware(self._clock())
        strategy = self._strategies[run.run_id]
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
        strategy.save_runtime_state(self._store)
        waiting = sum(signal.status is SignalStatus.WAITING_FOR_ENTRY for signal in evaluated)
        self._store.increment(run.run_id, session, "signals", waiting)
        ready_count = sum(row.status is Stage5Status.READY for row in valid_rows)
        self._store.increment(run.run_id, session, "instruments_ready", ready_count)
        context_not_ready = sum(
            row.status is Stage5Status.PRE_CONTEXT_NOT_READY for row in valid_rows
        )
        if finalize:
            self._store.increment(run.run_id, session, "checkpoints_processed")
            self._store.mark_checkpoint(
                run.run_id, session, t0, CheckpointState.COMPLETED,
                f"Stage 5 ready={ready_count}; candidate_errors={candidate_errors}", now,
            )
        self._logger.info(
            "checkpoint_batch_processed", run_id=run.run_id, t0=t0.isoformat(),
            checkpoint=checkpoint, instruments_ready=ready_count,
            pre_context_not_ready=context_not_ready, strategy_signals=waiting,
            candidate_errors=candidate_errors,
        )

    async def _observe_entries(self, now: datetime) -> None:
        # Register all baseline opportunities before resolving outcomes/admissions,
        # so pool availability cannot depend on the order of runs in this cycle.
        prepared = []
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
                now = _aware(self._clock())
                expired = strategy.expire_waiting_before(now)
                if expired:
                    self._store.save_signals(expired, now)
                observed_signals = strategy.signals
                waiting = tuple(
                    signal
                    for signal in observed_signals
                    if signal.status is SignalStatus.WAITING_FOR_ENTRY
                )
                if waiting:
                    before_observation = {s.signal_id: s for s in observed_signals}
                    source = self._services_for(run).entries
                    events = await source.trades_for(instruments, waiting)
                    for con_id, reason in source.trade_errors.items():
                        strategy.expire_unobservable(con_id, reason)
                    strategy.observe_trades(events)
                    observed_signals = strategy.signals
                    self._store.save_signals(
                        tuple(s for s in observed_signals
                              if s is not before_observation.get(s.signal_id)
                              and s != before_observation.get(s.signal_id)),
                        now,
                    )
                attempted = self._ledger.attempted_signal_ids(run.run_id)
                intents = tuple(
                    s
                    for s in observed_signals
                    if s.status is SignalStatus.ENTRY_TRIGGERED
                    and s.selected
                    and s.signal_id not in attempted
                )
                for signal in intents:
                    if is_baseline_payoff_candidate(signal):
                        instrument = (
                            instruments.get(signal.underlying_con_id)
                            if signal.underlying_con_id
                            else None
                        )
                        if instrument is not None:
                            self._store.register_payoff(
                                signal,
                                instrument,
                                run,
                                self._config.session_hard_hv_round_trip_cost_bps,
                            )
                prepared.append((run, market, strategy, strategy_runtime, instruments, intents))
            except Exception as exc:
                self._set_run(run.run_id, RunRuntimeState.DEGRADED, str(exc))
                self._logger.error("run_degraded", run_id=run.run_id, reason=str(exc))
                continue

        # Shadow tracking also runs for historical/removed universes and disabled
        # runs, using the original persisted qualified instrument and run identity.
        await self._advance_pending_payoffs(now, fetch_missing=False)
        # All runs have consumed this cycle's events. Keep streams only for waiting
        # candidates or checkpoints whose qualification has not yet completed.
        retained_by_source: dict[int, tuple[EntryBarSource, set[int]]] = {}
        for run, market, strategy, _runtime, instruments, _intents in prepared:
            source = self._services_for(run).entries
            retained = retained_by_source.setdefault(id(source), (source, set()))[1]
            retained.update(
                s.underlying_con_id
                for s in strategy.signals
                if s.status is SignalStatus.WAITING_FOR_ENTRY and s.underlying_con_id is not None
            )
            data = (run.method_spec or {}).get("data_requirements", {})
            entry = (run.method_spec or {}).get("entry", {})
            for _, t0 in self._services_for(run).checkpoints(market):
                if now < t0 <= now + timedelta(minutes=data.get("prefetch_lead_minutes", 0)) or (
                    t0 <= now < t0 + timedelta(minutes=entry.get("window_minutes", 0))
                    and self._store.checkpoint_state(run.run_id, market.session, t0)
                    in {None, CheckpointState.PROCESSING}
                ):
                    retained.update(instruments)
        for source, retained in retained_by_source.values():
            source.release_unused_trades(retained)
        ready_for_execution = []
        for run, market, strategy, strategy_runtime, instruments, intents in prepared:
            try:
                admitted: list[StrategySignal] = []
                if not intents:
                    ready_for_execution.append(
                        (run, market, strategy, strategy_runtime, instruments, admitted)
                    )
                    continue
                before_admission = {s.signal_id: s for s in strategy.signals}
                for signal in intents:
                    if is_baseline_payoff_candidate(signal):
                        if signal.underlying_con_id not in instruments:
                            continue
                        assessment = self._store.assess_payoff(signal.signal_id)
                        strategy.record_admission(signal.signal_id, assessment.decision.value)
                        self._logger.info(
                            "session_hard_cost_admission",
                            symbol=signal.symbol,
                            signal_id=signal.signal_id,
                            setup="HIGH_PRE_MOVE_DOWN",
                            signal_timestamp=str(signal.signal_timestamp),
                            **asdict(assessment),
                        )
                        if not assessment.take_trade:
                            continue
                    admitted.append(signal)
                self._store.save_signals(
                    tuple(
                        s for s in strategy.signals
                        if s is not before_admission.get(s.signal_id)
                        and s != before_admission.get(s.signal_id)
                    ),
                    now,
                )
                ready_for_execution.append(
                    (run, market, strategy, strategy_runtime, instruments, admitted)
                )
            except Exception as exc:
                self._set_run(run.run_id, RunRuntimeState.DEGRADED, str(exc))
                self._logger.error("run_degraded", run_id=run.run_id, reason=str(exc))

        # Freeze every simultaneous decision before broker awaits can allow a
        # background history completion to change the pool between runs.
        for run, market, _strategy, strategy_runtime, instruments, admitted in ready_for_execution:
            try:
                attempts = await strategy_runtime.execute_observed(admitted, instruments)
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
        # Historical downloads never stand between a ready signal and submission.
        minute = now.replace(second=0, microsecond=0)
        if (
            not self._stopping
            and self._payoff_fetch_minute != minute
            and (self._payoff_history_task is None or self._payoff_history_task.done())
        ):
            self._payoff_fetch_minute = minute
            self._payoff_history_task = asyncio.create_task(self._advance_background_history(now))

    async def _advance_background_history(self, now: datetime) -> None:
        for run_id, method in self._strategies.items():
            requests, _ = self._qualification_for({run_id})
            instruments = {r.instrument.con_id: r.instrument for r in requests}
            source = self._services_for(self._manager.get_run(run_id).config).entries
            await method.advance_runtime_state(source, instruments, self._store, now, self._logger)
        await self._advance_pending_payoffs(now, fetch_missing=True)

    async def _advance_pending_payoffs(
        self,
        now: datetime,
        *,
        fetch_missing: bool = False,
    ) -> None:
        pending_by_run: dict[str, list[tuple[StrategySignal, QualifiedInstrument, RunConfig]]] = {}
        for item in self._store.pending_payoffs():
            pending_by_run.setdefault(item[2].run_id, []).append(item)
        for items in pending_by_run.values():
            run = items[0][2]
            instruments = {item[1].con_id: item[1] for item in items}
            try:
                bars = await self._entry_source.bars_for(
                    run,
                    instruments,
                    session=items[0][0].session,
                    now=now,
                    signals=tuple(item[0] for item in items),
                    fetch_missing=fetch_missing,
                )
            except Exception as exc:
                self._logger.warning(
                    "session_hard_shadow_incomplete", run_id=run.run_id, reason=str(exc)
                )
                continue
            for signal, instrument, _run in items:
                try:
                    outcome = hypothetical_baseline_outcome(
                        signal,
                        bars.get(instrument.con_id, ()),
                        as_of=now,
                    )
                    if outcome is not None:
                        self._store.complete_payoff(signal.signal_id, outcome)
                        self._logger.info(
                            "session_hard_baseline_completed",
                            signal_id=signal.signal_id,
                            symbol=signal.symbol,
                            gross_r=outcome.gross_r,
                            completion_timestamp=outcome.completion_timestamp.isoformat(),
                        )
                except Exception as exc:
                    self._logger.warning(
                        "session_hard_shadow_incomplete",
                        signal_id=signal.signal_id,
                        symbol=signal.symbol,
                        reason=str(exc),
                    )
        self._store.record_payoff_executions(self._ledger.entry_fill_signal_ids())

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
    for name in ("entry_timestamp", "signal_timestamp", "armed_at", "deadline"):
        value = getattr(signal, name)
        payload[name] = _aware(value).isoformat() if value is not None else None
    return payload


def _signal_from_payload(payload: Mapping[str, object]) -> StrategySignal:
    values = dict(payload)
    values["session"] = date.fromisoformat(str(values["session"]))
    values["t0"] = datetime.fromisoformat(str(values["t0"]))
    values["status"] = SignalStatus(str(values["status"]))
    values["band"] = PreMoveBand(str(values["band"])) if values.get("band") else None
    for name in ("entry_timestamp", "signal_timestamp", "armed_at", "deadline"):
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
    data_clock = clock or (lambda: datetime.now(tz=UTC))
    snapshot_store = Stage5SnapshotStore(database_path)
    services = {
        method.version: create_method_services(
            method.method_id,
            method.version,
            market_data_broker,
            history_cache,
            snapshot_store,
            data_clock,
            logger,
        )
        for method in installed_methods()
    }
    default = next(iter(services.values()))

    async def qualify(runs_to_prepare: Sequence[RunInstance]) -> Stage5QualificationResult:
        results = []
        for version, service in services.items():
            selected = tuple(r for r in runs_to_prepare if r.config.strategy_version == version)
            if not selected:
                continue
            assert service.qualify is not None
            results.append(await service.qualify(selected))
        return _merge_qualification_results(*results)

    runtime_store = RuntimeStore(database_path)
    runtime_store.backfill_payoffs(runs.runs, history_cache)
    return StockerRuntime(
        config=runs,
        execution_router=execution_router,
        ledger=ExecutionLedger(database_path),
        store=runtime_store,
        qualify=qualify,
        stage5=default.features,
        stage5_by_strategy={version: service.features for version, service in services.items()},
        method_services=services,
        context_provider=default.context,
        entry_source=default.entries,
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
