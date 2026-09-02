import asyncio
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest

from stocker_core.config import IbkrConfig, RunsConfig
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
from stocker_execution.stage7 import ExecutionDestination, ExecutionRouter

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
        environment: Environment = Environment.PAPER,
    ) -> None:
        self.environment = environment
        self.configured_account = account
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
        self.account = self.account or self.configured_account
        self.is_connected = True
        self.connection_epoch += 1
        return BrokerSession(self.environment, self.account, True)

    def disconnect(self) -> None:
        self.events.append("disconnect")
        self.is_connected = False
        self.account = ""
        self.connection_epoch += 1

    def reconfigure(self, config: IbkrConfig) -> None:
        assert not self.is_connected
        assert config.environment is self.environment
        self.events.append("reconfigure")

    async def account_state(self) -> BrokerAccountState:
        self.events.append("account_state")
        return BrokerAccountState(
            self.environment,
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


class CapturingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **values: object) -> None:
        self.events.append((event, values))

    def warning(self, event: str, **values: object) -> None:
        self.events.append((event, values))

    def error(self, event: str, **values: object) -> None:
        self.events.append((event, values))


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


class RollingTriggerEntrySource:
    async def bars_for(
        self,
        run: RunConfig,
        instruments: object,
        *,
        session: date,
        now: datetime,
        signals: object,
    ) -> dict[int, tuple[EntryBar, ...]]:
        timestamp = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
        return {
            con_id: (EntryBar(timestamp, 99.5, 99.7, 99.4),)
            for con_id in instruments
        }


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
    qualification_log: list[tuple[str, ...]] | None = None,
    live_broker: FakeBroker | None = None,
    logger: object | None = None,
) -> StockerRuntime:
    selected_runs = runs or (_run(),)
    features = feature_service or FakeFeatureService()

    async def qualify(active_runs: object) -> Stage5QualificationResult:
        instances = tuple(active_runs)
        if qualification_log is not None:
            qualification_log.append(tuple(item.config.run_id for item in instances))
        memberships = tuple(
            Stage5Membership(instance.config.run_id, instance.config.universe)
            for instance in instances
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

    broker_arguments: dict[str, object]
    if live_broker is None:
        broker_arguments = {"broker": broker, "expected_account": "DU123456"}
    else:
        broker_arguments = {
            "execution_router": ExecutionRouter(
                (
                    ExecutionDestination(Environment.PAPER, "DU123456", broker),
                    ExecutionDestination(Environment.LIVE, "U123456", live_broker),
                )
            )
        }
    return StockerRuntime(
        config=_runs(*selected_runs),
        ledger=ExecutionLedger(tmp_path / "execution.sqlite3"),
        store=RuntimeStore(tmp_path / "runtime.sqlite3"),
        qualify=qualify,
        stage5=Stage5Analyzer(features),
        context_provider=context_provider or EmptyContextProvider(),
        entry_source=entry_source or EmptyEntrySource(),
        session_resolver=FixedSessionResolver(),
        clock=clock or MutableClock(),
        logger=logger,
        **broker_arguments,
    )


def test_clean_startup_reconciles_before_reaching_ready(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime = _runtime(tmp_path, broker)

    asyncio.run(runtime.start())

    status = runtime.status()
    assert status.application is ApplicationState.READY
    assert status.runs[0].state is RunRuntimeState.ACTIVE
    paper = status.execution_environments[0]
    assert paper.equity == 100_000.0
    assert paper.buying_power == 200_000.0
    assert broker.events.index("connect") < broker.events.index("open_orders")
    assert broker.events.index("open_orders") < broker.events.index("positions")
    assert "submit" not in broker.events

    broker.disconnect()
    disconnected = runtime.status().execution_environments[0]
    assert disconnected.equity is None
    assert disconnected.buying_power is None


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


def test_hot_enable_prepares_run_without_replaying_missed_checkpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = MutableClock()
        features = FakeFeatureService()
        runtime = _runtime(
            tmp_path,
            FakeBroker(),
            _run("existing"),
            _run("hot", enabled=False),
            clock=clock,
            feature_service=features,
            context_provider=TriggerContextProvider(),
        )
        await runtime.start()
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

        status = await runtime.apply_runs_config(
            _runs(_run("existing"), _run("hot", enabled=True)),
            frozenset({"hot"}),
        )
        await runtime.poll_once()

        states = {item.run_id: item.state for item in status.runs}
        assert states["hot"] is RunRuntimeState.ACTIVE
        assert features.calls == 1  # Existing run only; the newly enabled run is not replayed.
        assert (
            runtime.store.checkpoint_state(
                "hot", SESSION, datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
            )
            is not None
        )

    asyncio.run(scenario())


def test_hot_disable_stops_future_evaluation_without_touching_broker_exposure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        clock = MutableClock()
        features = FakeFeatureService()
        position = BrokerPosition("DU123456", 1000, "AAPL", -10, 100.0)
        broker = FakeBroker()
        runtime = _runtime(
            tmp_path,
            broker,
            clock=clock,
            feature_service=features,
            context_provider=TriggerContextProvider(),
        )
        await runtime.start()
        broker.positions = (position,)
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)

        status = await runtime.apply_runs_config(
            _runs(_run(enabled=False)),
            frozenset({"paper-run"}),
        )
        await runtime.poll_once()

        assert status.runs[0].state is RunRuntimeState.DISABLED
        assert features.calls == 0
        assert broker.positions == (position,)
        assert not hasattr(broker, "cancel_order")
        assert "disconnect" not in broker.events

    asyncio.run(scenario())


def test_hot_risk_change_drives_future_sizing_without_reconnecting(tmp_path: Path) -> None:
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
        changed = _run().model_copy(
            update={
                "risk": RunRiskConfig(
                    risk_per_trade=0.0005,
                    max_concurrent_positions=2,
                )
            }
        )

        await runtime.apply_runs_config(_runs(changed), frozenset({"paper-run"}))
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
        await runtime.poll_once()

        assert len(broker.submitted) == 1
        assert broker.submitted[0].quantity == 50
        assert "disconnect" not in broker.events

    asyncio.run(scenario())


def test_environment_hot_change_routes_only_future_orders_and_preserves_exposure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        clock = MutableClock()
        paper = FakeBroker()
        live = FakeBroker(account="U123456", environment=Environment.LIVE)
        runtime = _runtime(
            tmp_path,
            paper,
            _run("switching"),
            clock=clock,
            context_provider=TriggerContextProvider(),
            entry_source=RollingTriggerEntrySource(),
            live_broker=live,
        )
        await runtime.start()
        await runtime.replace_broker_config(
            IbkrConfig(
                environment=Environment.LIVE,
                host="live-gateway.local",
                port=4001,
                client_id=42,
                expected_account="U123456",
            )
        )
        clock.now = datetime(2026, 9, 2, 14, 1, tzinfo=UTC)
        await runtime.poll_once()
        paper_plan = paper.submitted[0]
        runtime.record_fill(
            BrokerFill(
                "paper-entry",
                101,
                "DU123456",
                Environment.PAPER,
                paper_plan.con_id,
                paper_plan.symbol,
                OrderAction.SELL,
                paper_plan.quantity,
                99.5,
                clock.now,
                0.0,
            )
        )
        paper.positions = (
            BrokerPosition(
                "DU123456", paper_plan.con_id, paper_plan.symbol, -paper_plan.quantity, 99.5
            ),
        )
        paper.open_orders = (
            BrokerOpenOrder(
                102,
                paper_plan.order_plan_id,
                "DU123456",
                Environment.PAPER,
                paper_plan.con_id,
                paper_plan.symbol,
                OrderRole.STOP,
                OrderLifecycle.SUBMITTED,
            ),
            BrokerOpenOrder(
                103,
                paper_plan.order_plan_id,
                "DU123456",
                Environment.PAPER,
                paper_plan.con_id,
                paper_plan.symbol,
                OrderRole.TARGET,
                OrderLifecycle.SUBMITTED,
            ),
        )
        clock.now = datetime(2026, 9, 2, 14, 2, tzinfo=UTC)

        live_status = await runtime.apply_runs_config(
            _runs(_run("switching", environment=Environment.LIVE)),
            frozenset({"switching"}),
        )
        clock.now = datetime(2026, 9, 2, 14, 11, tzinfo=UTC)
        await runtime.poll_once()

        record = ExecutionLedger(tmp_path / "execution.sqlite3").get(
            paper_plan.order_plan_id
        )
        assert record is not None and record.environment is Environment.PAPER
        assert live_status.runs[0].environment is Environment.LIVE
        assert len(paper.submitted) == 1
        assert len(live.submitted) == 1

        paper_status = await runtime.apply_runs_config(
            _runs(_run("switching", environment=Environment.PAPER)),
            frozenset({"switching"}),
        )
        record = ExecutionLedger(tmp_path / "execution.sqlite3").get(
            paper_plan.order_plan_id
        )
        assert record is not None and record.environment is Environment.PAPER
        assert paper_status.runs[0].environment is Environment.PAPER

    asyncio.run(scenario())


def test_run_loop_reconnects_legacy_environment_after_hot_change(tmp_path: Path) -> None:
    async def scenario() -> None:
        paper = FakeBroker()
        live = FakeBroker(account="U123456", environment=Environment.LIVE)
        runtime = _runtime(
            tmp_path,
            paper,
            _run("switching"),
            live_broker=live,
        )
        loop = asyncio.create_task(runtime.run_forever(poll_interval_seconds=0.001))
        try:
            for _ in range(200):
                if paper.is_connected:
                    break
                await asyncio.sleep(0.001)
            assert paper.is_connected
            await runtime.replace_broker_config(
                IbkrConfig(
                    environment=Environment.LIVE,
                    host="live-gateway.local",
                    port=4001,
                    client_id=42,
                    expected_account="U123456",
                )
            )
            await runtime.apply_runs_config(
                _runs(_run("switching", environment=Environment.LIVE)),
                frozenset({"switching"}),
            )
            initial_connects = paper.events.count("connect")

            paper.disconnect()
            for _ in range(200):
                if paper.events.count("connect") > initial_connects:
                    break
                await asyncio.sleep(0.001)

            assert paper.events.count("connect") == initial_connects + 1
            assert paper.is_connected
        finally:
            await runtime.stop()
            await asyncio.wait_for(loop, timeout=1.0)

    asyncio.run(scenario())


def test_custom_universe_change_reprepares_only_affected_run_without_stage5_fetch(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        calls: list[tuple[str, ...]] = []
        features = FakeFeatureService()
        runtime = _runtime(
            tmp_path,
            FakeBroker(),
            _run("custom", enabled=True).model_copy(update={"universe": "CUSTOM_US"}),
            _run("other"),
            feature_service=features,
            qualification_log=calls,
        )
        await runtime.start()
        calls.clear()
        current = _runs(
            _run("custom", enabled=True).model_copy(update={"universe": "CUSTOM_US"}),
            _run("other"),
        )
        custom = next(
            item for item in current.universes if item.universe_id == "CUSTOM_US"
        )
        updated_custom = custom.model_copy(
            update={
                "members": (
                    *custom.members,
                    InstrumentReference(
                        symbol="msft",
                        exchange="smart",
                        primary_exchange="nasdaq",
                        currency="usd",
                    ),
                )
            }
        )
        changed = RunsConfig(
            universes=tuple(
                updated_custom if item.universe_id == "CUSTOM_US" else item
                for item in current.universes
            ),
            runs=current.runs,
        )

        status = await runtime.apply_runs_config(changed, frozenset({"custom"}))

        assert next(item for item in status.runs if item.run_id == "custom").state is (
            RunRuntimeState.ACTIVE
        )
        assert calls == [("custom",)]
        assert features.calls == 0

    asyncio.run(scenario())


def test_custom_universe_change_batches_shared_instrument_preparation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        calls: list[tuple[str, ...]] = []
        custom_first = _run("custom-first").model_copy(
            update={"universe": "CUSTOM_US"}
        )
        custom_second = _run("custom-second").model_copy(
            update={"universe": "CUSTOM_US"}
        )
        runtime = _runtime(
            tmp_path,
            FakeBroker(),
            custom_first,
            custom_second,
            _run("other"),
            qualification_log=calls,
        )
        await runtime.start()
        calls.clear()
        current = _runs(custom_first, custom_second, _run("other"))
        custom = next(
            item for item in current.universes if item.universe_id == "CUSTOM_US"
        )
        changed = RunsConfig(
            universes=tuple(
                custom.model_copy(
                    update={
                        "members": (
                            *custom.members,
                            InstrumentReference(
                                symbol="MSFT",
                                exchange="SMART",
                                primary_exchange="NASDAQ",
                                currency="USD",
                            ),
                        )
                    }
                )
                if item.universe_id == "CUSTOM_US"
                else item
                for item in current.universes
            ),
            runs=current.runs,
        )

        await runtime.apply_runs_config(
            changed, frozenset({"custom-first", "custom-second"})
        )

        assert calls == [("custom-first", "custom-second")]

    asyncio.run(scenario())


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


def test_broker_config_change_reconnects_and_reconciles_affected_environment(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = FakeBroker()
        runtime = _runtime(tmp_path, broker)
        await runtime.start()
        broker.events.clear()

        status = await runtime.replace_broker_config(
            IbkrConfig(
                environment=Environment.PAPER,
                host="paper-gateway.local",
                port=4002,
                client_id=41,
                expected_account="DU123456",
            )
        )

        assert broker.events == [
            "disconnect",
            "reconfigure",
            "connect",
            "account_state",
            "open_orders",
            "statuses",
            "fills",
            "positions",
        ]
        paper = status.execution_environments[0]
        assert paper.environment is Environment.PAPER
        assert paper.connected is True
        assert paper.reconciled is True
        assert paper.ready is True

    asyncio.run(scenario())


def test_broker_config_changes_restart_only_the_selected_environment(tmp_path: Path) -> None:
    async def scenario() -> None:
        paper = FakeBroker()
        live = FakeBroker(account="U123456", environment=Environment.LIVE)
        runtime = _runtime(
            tmp_path,
            paper,
            _run("paper"),
            _run("live", environment=Environment.LIVE),
            live_broker=live,
        )
        await runtime.start()
        live_events = tuple(live.events)

        await runtime.replace_broker_config(
            IbkrConfig(
                environment=Environment.PAPER,
                host="paper-gateway.local",
                port=4002,
                client_id=41,
                expected_account="DU123456",
            )
        )

        assert tuple(live.events) == live_events
        assert live.is_connected is True

        paper_events = tuple(paper.events)
        await runtime.replace_broker_config(
            IbkrConfig(
                environment=Environment.LIVE,
                host="live-gateway.local",
                port=4001,
                client_id=42,
                expected_account="U123456",
            )
        )
        assert tuple(paper.events) == paper_events
        assert paper.is_connected is True

    asyncio.run(scenario())


def test_broker_account_change_with_existing_exposure_is_rejected_without_disconnect(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        broker = FakeBroker()
        runtime = _runtime(tmp_path, broker)
        await runtime.start()
        ExecutionLedger(tmp_path / "execution.sqlite3").replace_broker_snapshot(
            environment=Environment.PAPER,
            account="DU123456",
            positions=(BrokerPosition("DU123456", 1000, "AAPL", -10, 100.0),),
            open_orders=(),
            observed_at=NOW,
        )
        broker.events.clear()

        with pytest.raises(ValueError, match="while broker exposure exists"):
            await runtime.replace_broker_config(
                IbkrConfig(
                    environment=Environment.PAPER,
                    host="paper-gateway.local",
                    port=4002,
                    client_id=41,
                    expected_account="DU654321",
                )
            )

        assert broker.is_connected is True
        assert "disconnect" not in broker.events

    asyncio.run(scenario())


def test_deterministic_paper_flow_fills_position_and_protective_exit(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    broker = FakeBroker()
    logger = CapturingLogger()
    runtime = _runtime(
        tmp_path,
        broker,
        clock=clock,
        context_provider=TriggerContextProvider(),
        entry_source=TriggerEntrySource(),
        logger=logger,
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
    account_values = [
        values["account"]
        for _event, values in logger.events
        if values.get("account") is not None
    ]
    assert "DU***456" in account_values
    assert "DU123456" not in account_values


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
    assert "DU***456" in status.as_text()
    assert "DU123456" not in status.as_text()
    assert status.as_dict()["ibkr"]["PAPER"]["account"] == "DU***456"
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
