import asyncio
from datetime import UTC, date, datetime, time
from pathlib import Path

from stocker_core.config import RunsConfig
from stocker_core.runs import Environment, RunConfig, RunRiskConfig, RunWindow
from stocker_core.universes import InstrumentReference, UniverseDefinition
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerAccountState,
    BrokerFill,
    BrokerOpenOrder,
    BrokerOrderIds,
    BrokerOrderStatus,
    BrokerPosition,
    OrderAction,
    OrderLifecycle,
    OrderRole,
)
from stocker_execution.ibkr import BrokerSession, QualifiedInstrument
from stocker_execution.runtime import (
    ApplicationState,
    MarketSession,
    MarketSessionState,
    RunRuntimeState,
    RuntimeStore,
    StockerRuntime,
)
from stocker_execution.session_hard_structure_d import (
    SESSION_HARD_THRESHOLD,
    EntryBar,
    SessionHardAssessment,
    StrategyContext,
    StrategyOpportunityKey,
)
from stocker_execution.stage5 import (
    Stage5Analyzer,
    Stage5FeatureResult,
    Stage5Membership,
    Stage5QualificationResult,
    Stage5QualifiedRequest,
    Stage5Status,
)

NOW = datetime(2026, 9, 2, 13, 55, tzinfo=UTC)
SESSION = date(2026, 9, 2)


class FakeBroker:
    def __init__(
        self,
        *,
        account: str = "DU123456",
        connected: bool = False,
        connect_error: Exception | None = None,
        positions: tuple[BrokerPosition, ...] = (),
        open_orders: tuple[BrokerOpenOrder, ...] = (),
        fills: tuple[BrokerFill, ...] = (),
        statuses: tuple[BrokerOrderStatus, ...] = (),
    ) -> None:
        self.environment = Environment.PAPER
        self.account = account if connected else ""
        self.is_connected = connected
        self.connection_epoch = 1 if connected else 0
        self.connect_error = connect_error
        self.positions = positions
        self.open_orders = open_orders
        self.fills = fills
        self.statuses = statuses
        self.submitted: list[object] = []
        self.events: list[str] = []
        self.reconciliation_gate: asyncio.Event | None = None
        self.reconciliation_waiting: asyncio.Event | None = None

    async def connect(self) -> BrokerSession:
        self.events.append("connect")
        if self.connect_error is not None:
            raise self.connect_error
        self.account = self.account or "DU123456"
        self.is_connected = True
        self.connection_epoch += 1
        return BrokerSession(Environment.PAPER, self.account, True)

    def disconnect(self) -> None:
        self.events.append("disconnect")
        self.is_connected = False
        self.account = ""
        self.connection_epoch += 1

    async def account_state(self) -> BrokerAccountState:
        self.events.append("account_state")
        return BrokerAccountState(
            Environment.PAPER,
            self.account,
            100_000.0,
            200_000.0,
            self.is_connected,
            self.positions,
        )

    async def minimum_tick(self, instrument: QualifiedInstrument) -> float:
        return 0.01

    async def submit_protected_order(
        self, plan: object, instrument: QualifiedInstrument
    ) -> BrokerOrderIds:
        self.events.append("submit")
        self.submitted.append(plan)
        return BrokerOrderIds(101, 102, 103)

    async def read_open_orders(self) -> tuple[BrokerOpenOrder, ...]:
        self.events.append("open_orders")
        if self.reconciliation_gate is not None:
            if self.reconciliation_waiting is not None:
                self.reconciliation_waiting.set()
            await self.reconciliation_gate.wait()
        return self.open_orders

    async def read_fills(self) -> tuple[BrokerFill, ...]:
        self.events.append("fills")
        return self.fills

    async def read_positions(self) -> tuple[BrokerPosition, ...]:
        self.events.append("positions")
        return self.positions

    async def read_order_statuses(self) -> tuple[BrokerOrderStatus, ...]:
        self.events.append("statuses")
        return self.statuses


class FakeFeatureService:
    def __init__(self, *, session_offset: int = 0) -> None:
        self.calls = 0
        self.session_offset = session_offset

    async def get_feature(
        self, instrument: QualifiedInstrument, *, session: date, t0: datetime
    ) -> Stage5FeatureResult:
        self.calls += 1
        return Stage5FeatureResult(
            con_id=instrument.con_id,
            symbol=instrument.symbol,
            session=date.fromordinal(session.toordinal() + self.session_offset),
            t0=t0,
            status=Stage5Status.READY,
            exclusion_reason="",
            p0=100.0,
            expected_absolute_return_15m=0.02,
            m_price=2.0,
            raw_open_t0_minus_3m=100.8,
            raw_open_t0=100.0,
            alignment_factor=1.0,
            aligned_pre_open=100.8,
            raw_pre_move_price=0.8,
            pre_move_m=0.8,
        )


class BlockingFeatureService(FakeFeatureService):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_feature(
        self, instrument: QualifiedInstrument, *, session: date, t0: datetime
    ) -> Stage5FeatureResult:
        self.started.set()
        await self.release.wait()
        return await super().get_feature(instrument, session=session, t0=t0)


class FixedSessionResolver:
    def resolve(self, run: RunConfig, now: datetime) -> MarketSession:
        return MarketSession(
            session=SESSION,
            state=MarketSessionState.ACTIVE_SESSION,
            opens_at=datetime(2026, 9, 2, 13, 30, tzinfo=UTC),
            closes_at=datetime(2026, 9, 2, 20, 0, tzinfo=UTC),
        )


class EmptyContextProvider:
    async def context_for(
        self,
        run: RunConfig,
        rows: object,
        checkpoint: int,
        instruments: object,
        cohort_history: object,
    ) -> StrategyContext:
        return StrategyContext(run_id=run.run_id, session_hard={})


class EmptyEntrySource:
    async def bars_for(
        self,
        run: RunConfig,
        instruments: object,
        *,
        session: date,
        now: datetime,
        signals: object,
    ) -> dict[int, tuple[object, ...]]:
        return {}


class TriggerContextProvider:
    def __init__(self, *, failing_run: str | None = None) -> None:
        self.failing_run = failing_run

    async def context_for(
        self,
        run: RunConfig,
        rows: object,
        checkpoint: int,
        instruments: object,
        cohort_history: object,
    ) -> StrategyContext:
        if run.run_id == self.failing_run:
            raise RuntimeError("score inputs unavailable")
        snapshots = tuple(rows)
        return StrategyContext(
            run_id=run.run_id,
            session_hard={
                StrategyOpportunityKey(row.con_id, row.session, row.t0): SessionHardAssessment(
                    SESSION_HARD_THRESHOLD, checkpoint
                )
                for row in snapshots
                if row.con_id is not None
            },
        )


class TriggerEntrySource:
    async def bars_for(
        self,
        run: RunConfig,
        instruments: object,
        *,
        session: date,
        now: datetime,
        signals: object,
    ) -> dict[int, tuple[EntryBar, ...]]:
        t0 = datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
        return {con_id: (EntryBar(t0, 99.5, 99.7, 99.4),) for con_id in instruments}


class MutableClock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _runs(*runs: RunConfig) -> RunsConfig:
    universe_ids = {run.universe for run in runs}
    universes = tuple(
        UniverseDefinition(
            universe_id=universe_id,
            name=universe_id,
            members=(
                InstrumentReference(
                    symbol="AAPL", exchange="SMART", primary_exchange="NASDAQ", currency="USD"
                ),
            ),
        )
        for universe_id in sorted(universe_ids)
    )
    return RunsConfig(universes=universes, runs=runs)


def _run(
    run_id: str = "paper-run",
    *,
    environment: Environment = Environment.PAPER,
    enabled: bool = True,
) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        enabled=enabled,
        universe="NASDAQ",
        strategy="SESSION_HARD",
        environment=environment,
        risk=RunRiskConfig(risk_per_trade=0.001, max_concurrent_positions=5),
        session=RunWindow(
            start=time(9, 30),
            end=time(16),
            timezone="America/New_York",
            calendar="XNYS",
        ),
    )


def _runtime(
    tmp_path: Path,
    broker: FakeBroker,
    *runs: RunConfig,
    clock: MutableClock | None = None,
    feature_service: FakeFeatureService | None = None,
    context_provider: object | None = None,
    entry_source: object | None = None,
) -> StockerRuntime:
    selected_runs = runs or (_run(),)
    features = feature_service or FakeFeatureService()

    async def qualify(active_runs: object) -> Stage5QualificationResult:
        memberships = tuple(
            Stage5Membership(instance.config.run_id, instance.config.universe)
            for instance in active_runs
        )
        return Stage5QualificationResult(
            (
                Stage5QualifiedRequest(
                    QualifiedInstrument("AAPL", 1000, "SMART", "NASDAQ", "USD", "STK"),
                    memberships,
                ),
            ),
            (),
        )

    return StockerRuntime(
        config=_runs(*selected_runs),
        broker=broker,
        expected_account="DU123456",
        ledger=ExecutionLedger(tmp_path / "execution.sqlite3"),
        store=RuntimeStore(tmp_path / "runtime.sqlite3"),
        qualify=qualify,
        stage5=Stage5Analyzer(features),
        context_provider=context_provider or EmptyContextProvider(),
        entry_source=entry_source or EmptyEntrySource(),
        session_resolver=FixedSessionResolver(),
        clock=clock or MutableClock(),
    )


def test_clean_startup_reconciles_before_reaching_ready(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime = _runtime(tmp_path, broker)

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.READY
    assert runtime.status().runs[0].state is RunRuntimeState.ACTIVE
    assert broker.events.index("connect") < broker.events.index("open_orders")
    assert broker.events.index("open_orders") < broker.events.index("positions")
    assert "submit" not in broker.events


def test_wrong_account_prevents_readiness(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, FakeBroker(account="DU999999", connected=True))

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.DEGRADED
    assert runtime.status().runs[0].state is RunRuntimeState.DEGRADED
    assert "ACCOUNT" in runtime.status().runs[0].reason.upper()


def test_broker_unavailable_leaves_execution_safely_unavailable(tmp_path: Path) -> None:
    broker = FakeBroker(connect_error=RuntimeError("gateway unavailable"))
    runtime = _runtime(tmp_path, broker)

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.DEGRADED
    assert runtime.status().broker_connected is False
    assert broker.submitted == []


def test_enabled_run_requires_explicit_market_session_before_broker_connect(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    run = RunConfig(
        run_id="missing-session",
        universe="NASDAQ",
        strategy="SESSION_HARD",
        environment=Environment.PAPER,
        risk=RunRiskConfig(risk_per_trade=0.001, max_concurrent_positions=5),
    )
    runtime = _runtime(tmp_path, broker, run)

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.DEGRADED
    assert runtime.status().runs[0].reason == "explicit market session required"
    assert broker.events == []


def test_multiple_runs_coexist_and_disabled_run_never_activates(tmp_path: Path) -> None:
    runtime = _runtime(
        tmp_path,
        FakeBroker(),
        _run("first"),
        _run("second"),
        _run("disabled", enabled=False),
    )

    asyncio.run(runtime.start())

    states = {run.run_id: run.state for run in runtime.status().runs}
    assert states == {
        "first": RunRuntimeState.ACTIVE,
        "second": RunRuntimeState.ACTIVE,
        "disabled": RunRuntimeState.DISABLED,
    }


def test_unconfigured_live_environment_is_never_routed_to_paper(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime = _runtime(tmp_path, broker, _run("future-live", environment=Environment.LIVE))

    asyncio.run(runtime.start())

    status = runtime.status()
    assert status.application is ApplicationState.DEGRADED
    assert status.runs[0].state is RunRuntimeState.DEGRADED
    assert status.runs[0].reason == "EXECUTION_ENVIRONMENT_UNAVAILABLE"
    assert broker.events == []
    assert broker.submitted == []


def test_stop_blocks_new_work_and_preserves_broker_orders(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime = _runtime(tmp_path, broker)
    asyncio.run(runtime.start())

    asyncio.run(runtime.stop())

    assert runtime.status().application is ApplicationState.STOPPED
    assert runtime.status().runs[0].state is RunRuntimeState.STOPPED
    assert broker.events[-1] == "disconnect"
    assert not hasattr(broker, "cancel_order")


def test_stop_waits_for_in_flight_cycle_before_disconnect(tmp_path: Path) -> None:
    async def scenario() -> None:
        broker = FakeBroker()
        clock = MutableClock()
        features = BlockingFeatureService()
        runtime = _runtime(
            tmp_path,
            broker,
            clock=clock,
            feature_service=features,
            context_provider=TriggerContextProvider(),
        )
        await runtime.start()
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
        polling = asyncio.create_task(runtime.poll_once())
        await features.started.wait()
        stopping = asyncio.create_task(runtime.stop())
        await asyncio.sleep(0)

        assert broker.is_connected is True
        features.release.set()
        await polling
        await stopping
        assert broker.is_connected is False

    asyncio.run(scenario())


def test_checkpoint_is_evaluated_once_and_overlapping_runs_share_feature_work(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    features = FakeFeatureService()
    runtime = _runtime(
        tmp_path,
        FakeBroker(),
        _run("first"),
        _run("second"),
        clock=clock,
        feature_service=features,
        context_provider=TriggerContextProvider(),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())
    asyncio.run(runtime.poll_once())

    assert features.calls == 1
    assert runtime.status().counters.checkpoints_processed == 2


def test_late_start_marks_missed_checkpoint_without_evaluating(tmp_path: Path) -> None:
    clock = MutableClock(datetime(2026, 9, 2, 14, 1, tzinfo=UTC))
    features = FakeFeatureService()
    runtime = _runtime(
        tmp_path,
        FakeBroker(),
        clock=clock,
        feature_service=features,
        context_provider=TriggerContextProvider(),
    )

    asyncio.run(runtime.start())
    asyncio.run(runtime.poll_once())

    assert features.calls == 0
    assert (
        runtime.store.checkpoint_state(
            "paper-run", SESSION, datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
        ).value
        == "SKIPPED_MISSED"
    )


def test_one_run_context_failure_does_not_stop_unrelated_run(tmp_path: Path) -> None:
    clock = MutableClock()
    runtime = _runtime(
        tmp_path,
        FakeBroker(),
        _run("broken"),
        _run("healthy"),
        clock=clock,
        context_provider=TriggerContextProvider(failing_run="broken"),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())

    states = {run.run_id: run.state for run in runtime.status().runs}
    assert states["broken"] is RunRuntimeState.DEGRADED
    assert states["healthy"] is RunRuntimeState.ACTIVE


def test_session_mismatched_stage5_input_is_not_traded(tmp_path: Path) -> None:
    clock = MutableClock()
    broker = FakeBroker()
    runtime = _runtime(
        tmp_path,
        broker,
        clock=clock,
        feature_service=FakeFeatureService(session_offset=-1),
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())

    assert broker.submitted == []
    assert runtime.status().runs[0].state is RunRuntimeState.DEGRADED


def test_unexpected_position_blocks_affected_execution(tmp_path: Path) -> None:
    broker = FakeBroker(positions=(BrokerPosition("DU123456", 999, "UNKNOWN", 10, 50.0),))
    runtime = _runtime(tmp_path, broker)

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.DEGRADED
    assert "unexpected broker position" in runtime.status().runs[0].reason
    assert broker.submitted == []


def test_unexpected_order_blocks_affected_execution(tmp_path: Path) -> None:
    unknown = BrokerOpenOrder(
        999,
        "external-plan",
        "DU123456",
        Environment.PAPER,
        1000,
        "AAPL",
        OrderRole.ENTRY,
        OrderLifecycle.SUBMITTED,
    )
    runtime = _runtime(tmp_path, FakeBroker(open_orders=(unknown,)))

    asyncio.run(runtime.start())

    assert runtime.status().application is ApplicationState.DEGRADED
    assert "unexpected broker open order" in runtime.status().runs[0].reason


def test_disconnect_blocks_work_and_reconnect_requires_reconciliation(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = MutableClock()
        broker = FakeBroker()
        runtime = _runtime(
            tmp_path,
            broker,
            clock=clock,
            context_provider=TriggerContextProvider(),
            entry_source=TriggerEntrySource(),
        )
        await runtime.start()
        broker.is_connected = False
        broker.account = ""
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

        await runtime.poll_once()

        assert runtime.status().application is ApplicationState.DEGRADED
        assert broker.submitted == []
        gate = asyncio.Event()
        waiting = asyncio.Event()
        broker.reconciliation_gate = gate
        broker.reconciliation_waiting = waiting
        reconnecting = asyncio.create_task(runtime.reconnect())
        await waiting.wait()
        assert runtime.status().application is ApplicationState.RECONCILING
        await runtime.poll_once()
        assert broker.submitted == []
        gate.set()
        await reconnecting
        assert runtime.status().application is ApplicationState.READY
        assert broker.events[-4:] == ["open_orders", "statuses", "fills", "positions"]

    asyncio.run(scenario())


def test_reconnect_with_unknown_exposure_stays_degraded(tmp_path: Path) -> None:
    async def scenario() -> None:
        broker = FakeBroker()
        runtime = _runtime(tmp_path, broker)
        await runtime.start()
        broker.is_connected = False
        broker.account = ""
        await runtime.poll_once()
        broker.positions = (BrokerPosition("DU123456", 999, "UNKNOWN", 1, 10.0),)

        await runtime.reconnect()

        assert runtime.status().application is ApplicationState.DEGRADED
        assert "unexpected broker position" in runtime.status().runs[0].reason

    asyncio.run(scenario())


def test_deterministic_paper_flow_fills_position_and_protective_exit(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    broker = FakeBroker()
    runtime = _runtime(
        tmp_path,
        broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())

    assert len(broker.submitted) == 1
    plan = broker.submitted[0]
    entry = BrokerFill(
        "entry-fill",
        101,
        "DU123456",
        Environment.PAPER,
        1000,
        "AAPL",
        OrderAction.SELL,
        plan.quantity,
        99.5,
        datetime(2026, 9, 2, 14, 1, 10, tzinfo=UTC),
        1.0,
    )
    exit_fill = BrokerFill(
        "target-fill",
        103,
        "DU123456",
        Environment.PAPER,
        1000,
        "AAPL",
        OrderAction.BUY,
        plan.quantity,
        97.5,
        datetime(2026, 9, 2, 14, 4, tzinfo=UTC),
        1.0,
    )

    assert runtime.record_fill(entry) is True
    assert (
        len(
            ExecutionLedger(tmp_path / "execution.sqlite3").positions(Environment.PAPER, "DU123456")
        )
        == 1
    )
    assert runtime.record_fill(exit_fill) is True
    record = ExecutionLedger(tmp_path / "execution.sqlite3").get(plan.order_plan_id)
    assert record is not None
    assert record.status is OrderLifecycle.CLOSED
    assert record.realized_pnl == 2.0 * plan.quantity - 2.0
    assert (
        ExecutionLedger(tmp_path / "execution.sqlite3").positions(Environment.PAPER, "DU123456")
        == ()
    )
    assert runtime.status().counters.orders == 1
    assert runtime.status().counters.fills == 2


def test_duplicate_runtime_poll_does_not_duplicate_order(tmp_path: Path) -> None:
    clock = MutableClock()
    broker = FakeBroker()
    runtime = _runtime(
        tmp_path,
        broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())
    asyncio.run(runtime.poll_once())

    assert len(broker.submitted) == 1


def test_restart_restores_waiting_signal_without_reevaluating_checkpoint(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    first_broker = FakeBroker()
    first = _runtime(
        tmp_path,
        first_broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
    )
    asyncio.run(first.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
    asyncio.run(first.poll_once())
    asyncio.run(first.stop())
    assert first_broker.submitted == []

    second_broker = FakeBroker()
    restored = _runtime(
        tmp_path,
        second_broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(restored.start())
    clock.now = datetime(2026, 9, 2, 14, 2, tzinfo=UTC)
    asyncio.run(restored.poll_once())

    assert len(second_broker.submitted) == 1


def test_restart_does_not_replay_expired_waiting_signal(tmp_path: Path) -> None:
    clock = MutableClock()
    first = _runtime(
        tmp_path,
        FakeBroker(),
        clock=clock,
        context_provider=TriggerContextProvider(),
    )
    asyncio.run(first.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
    asyncio.run(first.poll_once())
    asyncio.run(first.stop())

    clock.now = datetime(2026, 9, 2, 14, 6, tzinfo=UTC)
    broker = FakeBroker()
    restored = _runtime(
        tmp_path,
        broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(restored.start())
    asyncio.run(restored.poll_once())

    assert broker.submitted == []


def _submitted_runtime(
    tmp_path: Path,
) -> tuple[StockerRuntime, FakeBroker, MutableClock, object]:
    clock = MutableClock()
    broker = FakeBroker()
    runtime = _runtime(
        tmp_path,
        broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
    )
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
    asyncio.run(runtime.poll_once())
    return runtime, broker, clock, broker.submitted[0]


def _open_order(plan: object, order_id: int, role: OrderRole) -> BrokerOpenOrder:
    return BrokerOpenOrder(
        order_id,
        plan.order_plan_id,
        "DU123456",
        Environment.PAPER,
        plan.con_id,
        plan.symbol,
        role,
        OrderLifecycle.SUBMITTED,
    )


def _fill(
    plan: object,
    *,
    execution_id: str,
    order_id: int,
    side: OrderAction,
    quantity: float,
    price: float,
) -> BrokerFill:
    return BrokerFill(
        execution_id,
        order_id,
        "DU123456",
        Environment.PAPER,
        plan.con_id,
        plan.symbol,
        side,
        quantity,
        price,
        datetime(2026, 9, 2, 14, 2, tzinfo=UTC),
        0.0,
        order_plan_id=plan.order_plan_id,
    )


def test_clean_restart_with_no_exposure_reaches_ready(tmp_path: Path) -> None:
    first = _runtime(tmp_path, FakeBroker())
    asyncio.run(first.start())
    asyncio.run(first.stop())
    restarted = _runtime(tmp_path, FakeBroker())

    asyncio.run(restarted.start())

    assert restarted.status().application is ApplicationState.READY


def test_restart_restores_known_pending_bracket(tmp_path: Path) -> None:
    first, _broker, _clock, plan = _submitted_runtime(tmp_path)
    asyncio.run(first.stop())
    orders = (
        _open_order(plan, 101, OrderRole.ENTRY),
        _open_order(plan, 102, OrderRole.STOP),
        _open_order(plan, 103, OrderRole.TARGET),
    )
    restarted = _runtime(tmp_path, FakeBroker(open_orders=orders))

    asyncio.run(restarted.start())

    assert restarted.status().application is ApplicationState.READY
    assert restarted.status().runs[0].open_positions == 0


def test_restart_restores_known_open_position_with_protection(tmp_path: Path) -> None:
    first, _broker, _clock, plan = _submitted_runtime(tmp_path)
    assert first.record_fill(
        _fill(
            plan,
            execution_id="entry",
            order_id=101,
            side=OrderAction.SELL,
            quantity=plan.quantity,
            price=99.5,
        )
    )
    asyncio.run(first.stop())
    restarted = _runtime(
        tmp_path,
        FakeBroker(
            positions=(BrokerPosition("DU123456", plan.con_id, plan.symbol, -plan.quantity, 99.5),),
            open_orders=(
                _open_order(plan, 102, OrderRole.STOP),
                _open_order(plan, 103, OrderRole.TARGET),
            ),
        ),
    )

    asyncio.run(restarted.start())

    assert restarted.status().application is ApplicationState.READY
    assert restarted.status().runs[0].open_positions == 1


def test_restart_restores_partially_filled_entry(tmp_path: Path) -> None:
    first, _broker, _clock, plan = _submitted_runtime(tmp_path)
    partial = plan.quantity / 2
    assert first.record_fill(
        _fill(
            plan,
            execution_id="partial-entry",
            order_id=101,
            side=OrderAction.SELL,
            quantity=partial,
            price=99.5,
        )
    )
    asyncio.run(first.stop())
    restarted = _runtime(
        tmp_path,
        FakeBroker(
            positions=(BrokerPosition("DU123456", plan.con_id, plan.symbol, -partial, 99.5),),
            open_orders=(
                _open_order(plan, 101, OrderRole.ENTRY),
                _open_order(plan, 102, OrderRole.STOP),
                _open_order(plan, 103, OrderRole.TARGET),
            ),
        ),
    )

    asyncio.run(restarted.start())

    record = ExecutionLedger(tmp_path / "execution.sqlite3").get(plan.order_plan_id)
    assert restarted.status().application is ApplicationState.READY
    assert record is not None
    assert record.status is OrderLifecycle.PARTIALLY_FILLED


def test_restart_ingests_trade_closed_while_offline(tmp_path: Path) -> None:
    first, _broker, _clock, plan = _submitted_runtime(tmp_path)
    asyncio.run(first.stop())
    restarted = _runtime(
        tmp_path,
        FakeBroker(
            fills=(
                _fill(
                    plan,
                    execution_id="offline-entry",
                    order_id=101,
                    side=OrderAction.SELL,
                    quantity=plan.quantity,
                    price=99.5,
                ),
                _fill(
                    plan,
                    execution_id="offline-target",
                    order_id=103,
                    side=OrderAction.BUY,
                    quantity=plan.quantity,
                    price=97.5,
                ),
            )
        ),
    )

    asyncio.run(restarted.start())

    record = ExecutionLedger(tmp_path / "execution.sqlite3").get(plan.order_plan_id)
    assert restarted.status().application is ApplicationState.READY
    assert record is not None
    assert record.status is OrderLifecycle.CLOSED


def test_restart_ingests_broker_rejection_while_offline(tmp_path: Path) -> None:
    first, _broker, _clock, plan = _submitted_runtime(tmp_path)
    asyncio.run(first.stop())
    rejection = BrokerOrderStatus(
        101,
        plan.order_plan_id,
        "DU123456",
        Environment.PAPER,
        OrderLifecycle.REJECTED,
        0.0,
        plan.quantity,
        "rejected while offline",
    )
    restarted = _runtime(tmp_path, FakeBroker(statuses=(rejection,)))

    asyncio.run(restarted.start())

    record = ExecutionLedger(tmp_path / "execution.sqlite3").get(plan.order_plan_id)
    assert restarted.status().application is ApplicationState.READY
    assert record is not None
    assert record.status is OrderLifecycle.REJECTED


def test_interrupted_checkpoint_is_not_replayed_after_restart(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    t0 = datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
    assert store.reserve_checkpoint("paper-run", SESSION, t0, 6, NOW)
    restarted = _runtime(tmp_path, FakeBroker())

    asyncio.run(restarted.start())

    assert restarted.store.checkpoint_state("paper-run", SESSION, t0) is not None
    assert restarted.store.checkpoint_state("paper-run", SESSION, t0).value == "SKIPPED_INTERRUPTED"


def test_status_has_text_and_json_ready_for_future_dashboard(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, FakeBroker())
    asyncio.run(runtime.start())

    status = runtime.status()

    assert "Application: READY" in status.as_text()
    assert "IBKR PAPER: connected" in status.as_text()
    assert status.as_dict()["runs"][0]["environment"] == "PAPER"


def test_no_signal_is_not_an_operational_failure(tmp_path: Path) -> None:
    clock = MutableClock()
    runtime = _runtime(tmp_path, FakeBroker(), clock=clock)
    asyncio.run(runtime.start())
    clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

    asyncio.run(runtime.poll_once())

    assert runtime.status().application is ApplicationState.READY
    assert runtime.status().runs[0].state is RunRuntimeState.ACTIVE
    assert runtime.status().runs[0].signals_today == 0
