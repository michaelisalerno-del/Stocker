import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from stocker_core.runs import Environment
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerAccountState,
    BrokerFill,
    BrokerOpenOrder,
    BrokerPosition,
    OrderLifecycle,
    OrderRole,
)
from stocker_execution.ibkr import BrokerSession, IbkrConnection, IbkrError
from stocker_execution.runtime import (
    ApplicationState,
    CheckpointState,
    RunRuntimeState,
    RuntimeStore,
    StockerRuntime,
    build_runtime,
)
from stocker_execution.session_hard_structure_d import (
    SESSION_HARD_THRESHOLD,
    EntryBar,
    SessionHardStructureDStrategy,
)
from stocker_execution.stage5 import (
    Stage5Analyzer,
    Stage5Membership,
    Stage5QualificationResult,
    Stage5QualifiedRequest,
    qualify_active_runs,
)
from stocker_execution.stage7 import (
    ExecutionDestination,
    ExecutionEnvironmentUnavailableError,
    ExecutionResultCode,
    ExecutionRouter,
    Stage7ExecutionService,
    Stage7RiskEngine,
    build_order_plan,
)
from test_stage5_pipeline import QualificationBoundary, active_run
from test_stage6_session_hard_strategy import ready_snapshot, strategy_context
from test_stage7_execution import FakeExecutionBroker, _instrument, _intent, _run
from test_stage7_ibkr import FakeOrderClient, _config, _plan
from test_stage8_runtime import (
    EmptyContextProvider,
    EmptyEntrySource,
    FakeBroker,
    FakeFeatureService,
    FixedSessionResolver,
    MutableClock,
    _runs,
)
from test_stage8_runtime import (
    _run as _runtime_run,
)


class EnvironmentBroker(FakeBroker):
    def __init__(self, environment: Environment, account: str) -> None:
        super().__init__(account=account)
        self.environment = environment

    async def connect(self) -> BrokerSession:
        self.events.append("connect")
        if self.connect_error is not None:
            raise self.connect_error
        self.account = self.account or (
            "DU123456" if self.environment is Environment.PAPER else "U123456"
        )
        self.is_connected = True
        self.connection_epoch += 1
        return BrokerSession(self.environment, self.account, True)

    async def account_state(self) -> BrokerAccountState:
        state = await super().account_state()
        return state.__class__(
            self.environment,
            state.account,
            state.equity,
            state.buying_power,
            state.connected,
            state.positions,
        )


class EquityExecutionBroker(FakeExecutionBroker):
    def __init__(self, environment: Environment, account: str, equity: float) -> None:
        super().__init__(environment=environment, account=account)
        self.equity = equity

    async def account_state(self) -> BrokerAccountState:
        state = await super().account_state()
        return replace(state, equity=self.equity)


def _mixed_runtime(
    tmp_path: Path,
    paper: EnvironmentBroker,
    live: EnvironmentBroker,
) -> StockerRuntime:
    paper_run = _runtime_run("paper-run")
    live_run = _runtime_run("live-run", environment=Environment.LIVE)
    router = ExecutionRouter(
        (
            ExecutionDestination(Environment.PAPER, "DU123456", paper),
            ExecutionDestination(Environment.LIVE, "U123456", live),
        )
    )

    async def qualify(active_runs):
        memberships = tuple(
            Stage5Membership(instance.config.run_id, instance.config.universe)
            for instance in active_runs
        )
        return Stage5QualificationResult(
            (Stage5QualifiedRequest(_instrument(), memberships),),
            (),
        )

    return StockerRuntime(
        config=_runs(paper_run, live_run),
        execution_router=router,
        ledger=ExecutionLedger(tmp_path / "execution.sqlite3"),
        store=RuntimeStore(tmp_path / "runtime.sqlite3"),
        qualify=qualify,
        stage5=Stage5Analyzer(FakeFeatureService()),
        context_provider=EmptyContextProvider(),
        entry_source=EmptyEntrySource(),
        session_resolver=FixedSessionResolver(),
        clock=MutableClock(),
    )


def test_execution_router_resolves_only_the_requested_environment() -> None:
    paper_broker = FakeExecutionBroker()
    live_broker = FakeExecutionBroker(environment=Environment.LIVE, account="U123456")
    router = ExecutionRouter(
        (
            ExecutionDestination(Environment.PAPER, "DU123456", paper_broker),
            ExecutionDestination(Environment.LIVE, "U123456", live_broker),
        )
    )

    assert router.for_environment(Environment.PAPER).broker is paper_broker
    assert router.for_environment(Environment.LIVE).broker is live_broker
    assert _run(Environment.LIVE).execution_environment is Environment.LIVE


def test_execution_router_never_falls_back_to_an_available_environment() -> None:
    router = ExecutionRouter(
        (
            ExecutionDestination(
                Environment.PAPER,
                "DU123456",
                FakeExecutionBroker(),
            ),
        )
    )

    try:
        router.for_environment(Environment.LIVE)
    except ExecutionEnvironmentUnavailableError as exc:
        assert str(exc) == "EXECUTION_ENVIRONMENT_UNAVAILABLE: LIVE"
    else:
        raise AssertionError("LIVE must not fall back to PAPER")


def test_live_run_reconciles_and_submits_through_the_same_execution_service(tmp_path) -> None:
    broker = FakeExecutionBroker(environment=Environment.LIVE, account="U123456")
    service = Stage7ExecutionService(
        run=_run(Environment.LIVE),
        expected_account="U123456",
        broker=broker,
        ledger=ExecutionLedger(tmp_path / "ledger.sqlite3"),
    )

    assert service.run_environment is Environment.LIVE
    assert asyncio.run(service.reconcile()).ok
    result = asyncio.run(service.execute(_intent(), _instrument()))

    assert result.code is ExecutionResultCode.SUBMITTED
    assert result.environment is Environment.LIVE
    assert result.actual_account == "U123456"
    assert result.order_plan is not None
    assert result.order_plan.environment is Environment.LIVE
    assert len(broker.submitted) == 1

    duplicate = asyncio.run(service.execute(_intent(), _instrument()))
    assert duplicate.code is ExecutionResultCode.DUPLICATE_ORDER_BLOCKED
    assert len(broker.submitted) == 1


def test_explicitly_enabled_live_connection_transmits_the_same_protected_bracket() -> None:
    client = FakeOrderClient(account="U123456")
    connection = IbkrConnection(
        _config(Environment.LIVE),
        client=client,
        execution_enabled=True,
    )

    async def scenario() -> None:
        await connection.connect()
        await connection.submit_protected_order(_plan(Environment.LIVE), _instrument())

    asyncio.run(scenario())

    assert client.connect_kwargs["readonly"] is False
    assert [order.orderType for _, order in client.placed] == ["MKT", "LMT", "STP"]
    assert {order.account for _, order in client.placed} == {"U123456"}


def test_ibkr_connection_rejects_a_cross_environment_plan_before_transmission() -> None:
    client = FakeOrderClient()
    connection = IbkrConnection(_config(Environment.PAPER), client=client, execution_enabled=True)

    async def scenario() -> None:
        await connection.connect()
        await connection.submit_protected_order(_plan(Environment.LIVE), _instrument())

    try:
        asyncio.run(scenario())
    except IbkrError as exc:
        assert str(exc) == "ACCOUNT_OR_ENVIRONMENT_MISMATCH"
    else:
        raise AssertionError("PAPER connection must reject a LIVE plan")
    assert client.placed == []


def test_live_ibkr_connection_rejects_a_paper_plan_before_transmission() -> None:
    client = FakeOrderClient(account="U123456")
    connection = IbkrConnection(_config(Environment.LIVE), client=client, execution_enabled=True)

    async def scenario() -> None:
        await connection.connect()
        await connection.submit_protected_order(_plan(Environment.PAPER), _instrument())

    try:
        asyncio.run(scenario())
    except IbkrError as exc:
        assert str(exc) == "ACCOUNT_OR_ENVIRONMENT_MISMATCH"
    else:
        raise AssertionError("LIVE connection must reject a PAPER plan")
    assert client.placed == []


def test_wrong_live_account_blocks_reconciliation_and_transmission(tmp_path) -> None:
    broker = FakeExecutionBroker(environment=Environment.LIVE, account="U-WRONG")
    service = Stage7ExecutionService(
        run=_run(Environment.LIVE),
        expected_account="U123456",
        broker=broker,
        ledger=ExecutionLedger(tmp_path / "ledger.sqlite3"),
    )

    reconciliation = asyncio.run(service.reconcile())
    result = asyncio.run(service.execute(_intent(), _instrument()))

    assert reconciliation.ok is False
    assert result.code is ExecutionResultCode.ACCOUNT_OR_ENVIRONMENT_MISMATCH
    assert broker.submitted == []


def test_market_data_qualification_is_shared_across_execution_environments() -> None:
    boundary = QualificationBoundary()
    paper = active_run("paper-run", "NASDAQ", "AAPL")
    live = replace(
        active_run("live-run", "NASDAQ", "AAPL"),
        config=active_run("live-run", "NASDAQ", "AAPL").config.model_copy(
            update={"environment": Environment.LIVE}
        ),
    )

    result = asyncio.run(qualify_active_runs(boundary, (paper, live)))

    assert boundary.calls == ["AAPL"]
    assert set(result.requests[0].memberships) == {
        Stage5Membership("paper-run", "NASDAQ"),
        Stage5Membership("live-run", "NASDAQ"),
    }
    assert result.ineligible == ()


def test_mixed_runtime_connects_reconciles_and_reports_each_environment(tmp_path) -> None:
    paper = EnvironmentBroker(Environment.PAPER, "DU123456")
    live = EnvironmentBroker(Environment.LIVE, "U123456")
    runtime = _mixed_runtime(tmp_path, paper, live)

    asyncio.run(runtime.start())
    status = runtime.status()

    assert status.application is ApplicationState.READY
    assert {run.run_id for run in status.runs if run.state.value == "ACTIVE"} == {
        "paper-run",
        "live-run",
    }
    assert {item.environment for item in status.execution_environments if item.ready} == {
        Environment.PAPER,
        Environment.LIVE,
    }
    assert paper.events[:1] == ["connect"]
    assert live.events[:1] == ["connect"]


def test_mixed_runtime_dispatches_each_signal_to_its_selected_environment(tmp_path) -> None:
    paper = EnvironmentBroker(Environment.PAPER, "DU123456")
    live = EnvironmentBroker(Environment.LIVE, "U123456")
    runtime = _mixed_runtime(tmp_path, paper, live)
    asyncio.run(runtime.start())

    paper_result = asyncio.run(runtime._execution["paper-run"].execute(_intent(), _instrument()))
    live_intent = replace(_intent(), run_id="live-run", signal_id="live-signal")
    live_result = asyncio.run(runtime._execution["live-run"].execute(live_intent, _instrument()))

    assert paper_result.code is ExecutionResultCode.SUBMITTED
    assert live_result.code is ExecutionResultCode.SUBMITTED
    assert len(paper.submitted) == 1
    assert len(live.submitted) == 1
    assert paper.submitted[0].environment is Environment.PAPER
    assert live.submitted[0].environment is Environment.LIVE


def test_same_instrument_uses_isolated_paper_and_live_account_state(tmp_path) -> None:
    paper = EquityExecutionBroker(Environment.PAPER, "DU123456", 100_000.0)
    live = EquityExecutionBroker(Environment.LIVE, "U123456", 50_000.0)
    ledger = ExecutionLedger(tmp_path / "ledger.sqlite3")
    paper_service = Stage7ExecutionService(
        run=_run(),
        expected_account="DU123456",
        broker=paper,
        ledger=ledger,
    )
    live_service = Stage7ExecutionService(
        run=_run(Environment.LIVE).model_copy(update={"run_id": "live-run"}),
        expected_account="U123456",
        broker=live,
        ledger=ledger,
    )
    live_intent = replace(_intent(), run_id="live-run", signal_id="live-signal-1")

    assert asyncio.run(paper_service.reconcile()).ok
    assert asyncio.run(live_service.reconcile()).ok
    paper_result = asyncio.run(paper_service.execute(_intent(), _instrument()))
    live_result = asyncio.run(live_service.execute(live_intent, _instrument()))

    assert paper_result.order_plan is not None
    assert live_result.order_plan is not None
    assert paper_result.order_plan.con_id == live_result.order_plan.con_id == 265598
    assert paper_result.order_plan.quantity == 100
    assert live_result.order_plan.quantity == 50
    assert paper_result.actual_account == "DU123456"
    assert live_result.actual_account == "U123456"
    assert ledger.known_order_ids(Environment.PAPER, "DU123456") == {101, 102, 103}
    assert ledger.known_order_ids(Environment.LIVE, "U123456") == {101, 102, 103}

    executed_at = datetime(2026, 9, 2, 14, 32, tzinfo=UTC)
    assert ledger.record_fill(
        BrokerFill(
            "same-broker-execution-id",
            101,
            "DU123456",
            Environment.PAPER,
            265598,
            "AAPL",
            paper_result.order_plan.side,
            10,
            100.0,
            executed_at,
        )
    )
    assert ledger.record_fill(
        BrokerFill(
            "same-broker-execution-id",
            101,
            "U123456",
            Environment.LIVE,
            265598,
            "AAPL",
            live_result.order_plan.side,
            10,
            100.0,
            executed_at,
        )
    )
    assert ledger.get(paper_result.order_plan.order_plan_id).filled_quantity == 10  # type: ignore[union-attr]
    assert ledger.get(live_result.order_plan.order_plan_id).filled_quantity == 10  # type: ignore[union-attr]


def test_risk_arithmetic_and_order_geometry_are_environment_independent() -> None:
    intent = _intent()
    paper_state = BrokerAccountState(Environment.PAPER, "DU123456", 100_000.0, 200_000.0, True, ())
    live_state = replace(paper_state, environment=Environment.LIVE, account="U123456")
    engine = Stage7RiskEngine()

    paper_risk = engine.evaluate(
        order_intent=intent,
        account_state=paper_state,
        risk_config=_run().risk,
    )
    live_risk = engine.evaluate(
        order_intent=intent,
        account_state=live_state,
        risk_config=_run(Environment.LIVE).risk,
    )
    created_at = datetime(2026, 9, 2, 14, 31, tzinfo=UTC)
    paper_plan = build_order_plan(
        order_intent=intent,
        risk_decision=paper_risk,
        environment=Environment.PAPER,
        minimum_tick=0.01,
        created_at=created_at,
    )
    live_plan = build_order_plan(
        order_intent=intent,
        risk_decision=live_risk,
        environment=Environment.LIVE,
        minimum_tick=0.01,
        created_at=created_at,
    )

    assert paper_risk == live_risk
    assert (
        paper_plan.side,
        paper_plan.quantity,
        paper_plan.entry_order_type,
        paper_plan.entry_reference,
        paper_plan.stop_price,
        paper_plan.target_price,
    ) == (
        live_plan.side,
        live_plan.quantity,
        live_plan.entry_order_type,
        live_plan.entry_reference,
        live_plan.stop_price,
        live_plan.target_price,
    )
    assert paper_plan.order_plan_id != live_plan.order_plan_id


def test_strategy_result_and_order_intent_are_environment_independent() -> None:
    paper_snapshot = replace(ready_snapshot(pre_move_m=0.8), run_ids=("paper-run",))
    live_snapshot = replace(paper_snapshot, run_ids=("live-run",))
    paper_strategy = SessionHardStructureDStrategy()
    live_strategy = SessionHardStructureDStrategy()
    paper_strategy.evaluate(
        (paper_snapshot,),
        strategy_context(
            (paper_snapshot,),
            {101: SESSION_HARD_THRESHOLD},
            run_id="paper-run",
        ),
    )
    live_strategy.evaluate(
        (live_snapshot,),
        strategy_context(
            (live_snapshot,),
            {101: SESSION_HARD_THRESHOLD},
            run_id="live-run",
        ),
    )
    entry_bar = EntryBar(
        paper_snapshot.t0,
        open=100.0,
        high=100.1,
        low=99.7,
    )

    paper_intent = paper_strategy.observe_entry_bars({101: (entry_bar,)})[0]
    live_intent = live_strategy.observe_entry_bars({101: (entry_bar,)})[0]

    assert (
        paper_intent.status,
        paper_intent.reason,
        paper_intent.side,
        paper_intent.direction,
        paper_intent.entry_level,
        paper_intent.entry_reference,
        paper_intent.stop_distance_m,
        paper_intent.target_distance_m,
        paper_intent.selected,
    ) == (
        live_intent.status,
        live_intent.reason,
        live_intent.side,
        live_intent.direction,
        live_intent.entry_level,
        live_intent.entry_reference,
        live_intent.stop_distance_m,
        live_intent.target_distance_m,
        live_intent.selected,
    )
    assert paper_intent.run_id == "paper-run"
    assert live_intent.run_id == "live-run"
    assert paper_intent.signal_id != live_intent.signal_id


def test_unknown_live_exposure_blocks_only_live_environment(tmp_path) -> None:
    paper = EnvironmentBroker(Environment.PAPER, "DU123456")
    live = EnvironmentBroker(Environment.LIVE, "U123456")
    live.positions = (BrokerPosition("U123456", 999, "UNKNOWN", 10, 50.0),)
    runtime = _mixed_runtime(tmp_path, paper, live)

    asyncio.run(runtime.start())
    status = runtime.status()
    runs = {run.run_id: run for run in status.runs}
    environments = {item.environment: item for item in status.execution_environments}

    assert status.application is ApplicationState.READY
    assert runs["paper-run"].state.value == "ACTIVE"
    assert runs["live-run"].state.value == "DEGRADED"
    assert "unexpected broker position" in runs["live-run"].reason
    assert environments[Environment.PAPER].ready is True
    assert environments[Environment.LIVE].ready is False


def test_unknown_paper_exposure_blocks_only_paper_environment(tmp_path) -> None:
    paper = EnvironmentBroker(Environment.PAPER, "DU123456")
    live = EnvironmentBroker(Environment.LIVE, "U123456")
    paper.positions = (BrokerPosition("DU123456", 999, "UNKNOWN", 10, 50.0),)
    runtime = _mixed_runtime(tmp_path, paper, live)

    asyncio.run(runtime.start())
    status = runtime.status()
    runs = {run.run_id: run for run in status.runs}
    environments = {item.environment: item for item in status.execution_environments}

    assert status.application is ApplicationState.READY
    assert runs["paper-run"].state.value == "DEGRADED"
    assert runs["live-run"].state.value == "ACTIVE"
    assert environments[Environment.PAPER].ready is False
    assert environments[Environment.LIVE].ready is True


def test_order_and_position_records_never_reconcile_across_environments(tmp_path) -> None:
    cases = (
        (Environment.PAPER, "DU123456", Environment.LIVE, "U123456"),
        (Environment.LIVE, "U123456", Environment.PAPER, "DU123456"),
    )
    for index, (
        source_environment,
        source_account,
        target_environment,
        target_account,
    ) in enumerate(cases):
        ledger = ExecutionLedger(tmp_path / f"ledger-{index}.sqlite3")
        source_run = _run(source_environment).model_copy(update={"run_id": f"source-{index}"})
        source_intent = replace(
            _intent(), run_id=source_run.run_id, signal_id=f"source-signal-{index}"
        )
        source_broker = FakeExecutionBroker(environment=source_environment, account=source_account)
        source = Stage7ExecutionService(
            run=source_run,
            expected_account=source_account,
            broker=source_broker,
            ledger=ledger,
        )
        assert asyncio.run(source.reconcile()).ok
        submitted = asyncio.run(source.execute(source_intent, _instrument()))
        assert submitted.order_plan is not None

        target_broker = FakeExecutionBroker(
            environment=target_environment,
            account=target_account,
            open_orders=(
                BrokerOpenOrder(
                    101,
                    submitted.order_plan.order_plan_id,
                    target_account,
                    target_environment,
                    265598,
                    "AAPL",
                    OrderRole.ENTRY,
                    OrderLifecycle.SUBMITTED,
                ),
            ),
            positions=(BrokerPosition(target_account, 265598, "AAPL", -10, 100.0),),
        )
        target = Stage7ExecutionService(
            run=_run(target_environment).model_copy(update={"run_id": f"target-{index}"}),
            expected_account=target_account,
            broker=target_broker,
            ledger=ledger,
        )

        reconciliation = asyncio.run(target.reconcile())

        assert reconciliation.ok is False
        assert reconciliation.code is ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED


def test_live_disconnect_blocks_only_live_environment(tmp_path) -> None:
    paper = EnvironmentBroker(Environment.PAPER, "DU123456")
    live = EnvironmentBroker(Environment.LIVE, "U123456")
    runtime = _mixed_runtime(tmp_path, paper, live)
    asyncio.run(runtime.start())
    live.is_connected = False
    live.account = ""

    asyncio.run(runtime.poll_once())
    status = runtime.status()
    runs = {run.run_id: run for run in status.runs}

    assert status.application is ApplicationState.READY
    assert runs["paper-run"].state.value == "ACTIVE"
    assert runs["live-run"].reason == "BROKER_DISCONNECTED"
    assert paper.is_connected is True


def test_live_reconnect_does_not_cycle_paper_and_requires_live_reconciliation(tmp_path) -> None:
    paper = EnvironmentBroker(Environment.PAPER, "DU123456")
    live = EnvironmentBroker(Environment.LIVE, "U123456")
    runtime = _mixed_runtime(tmp_path, paper, live)
    asyncio.run(runtime.start())
    paper_events = tuple(paper.events)
    live.is_connected = False
    live.account = ""
    asyncio.run(runtime.poll_once())

    asyncio.run(runtime.reconnect(Environment.LIVE))
    status = runtime.status()
    environments = {item.environment: item for item in status.execution_environments}

    assert tuple(paper.events) == paper_events
    assert environments[Environment.PAPER].ready is True
    assert environments[Environment.LIVE].reconciled is True
    assert environments[Environment.LIVE].ready is True


def test_live_reconnect_marks_only_live_missed_checkpoints(tmp_path) -> None:
    paper = EnvironmentBroker(Environment.PAPER, "DU123456")
    live = EnvironmentBroker(Environment.LIVE, "U123456")
    runtime = _mixed_runtime(tmp_path, paper, live)
    asyncio.run(runtime.start())
    live.is_connected = False
    live.account = ""
    asyncio.run(runtime.poll_once())
    runtime._clock = MutableClock(datetime(2026, 9, 2, 14, 6, tzinfo=UTC))

    asyncio.run(runtime.reconnect(Environment.LIVE))

    checkpoint = datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
    assert runtime.store.checkpoint_state("paper-run", checkpoint.date(), checkpoint) is None
    assert (
        runtime.store.checkpoint_state("live-run", checkpoint.date(), checkpoint)
        is CheckpointState.SKIPPED_MISSED
    )


def test_live_reconnect_preparation_failure_does_not_degrade_paper(tmp_path) -> None:
    paper = EnvironmentBroker(Environment.PAPER, "DU123456")
    live = EnvironmentBroker(Environment.LIVE, "U123456")
    runtime = _mixed_runtime(tmp_path, paper, live)
    asyncio.run(runtime.start())
    live.is_connected = False
    live.account = ""
    asyncio.run(runtime.poll_once())

    async def fail_preparation(_active_runs):
        raise RuntimeError("LIVE preparation failed")

    runtime._qualify = fail_preparation
    asyncio.run(runtime.reconnect(Environment.LIVE))
    status = runtime.status()
    runs = {run.run_id: run for run in status.runs}
    environments = {item.environment: item for item in status.execution_environments}

    assert status.application is ApplicationState.READY
    assert runs["paper-run"].state.value == "ACTIVE"
    assert environments[Environment.PAPER].ready is True
    assert runs["live-run"].state.value == "DEGRADED"
    assert environments[Environment.LIVE].ready is False


def test_live_readiness_diagnostic_reads_state_without_submitting_an_order(tmp_path) -> None:
    paper = EnvironmentBroker(Environment.PAPER, "DU123456")
    live = EnvironmentBroker(Environment.LIVE, "U123456")
    runtime = _mixed_runtime(tmp_path, paper, live)
    asyncio.run(runtime.start())

    diagnostic = asyncio.run(runtime.execution_readiness_diagnostic(Environment.LIVE))

    assert diagnostic.environment is Environment.LIVE
    assert diagnostic.connected is True
    assert diagnostic.account == "U123456"
    assert diagnostic.expected_account_match is True
    assert diagnostic.account_state_available is True
    assert diagnostic.open_orders == 0
    assert diagnostic.positions == 0
    assert diagnostic.reconciled is True
    assert diagnostic.ready is True
    assert live.submitted == []


def test_production_composition_loads_only_enabled_run_environments(tmp_path) -> None:
    runs_path = tmp_path / "runs.yaml"
    runs_path.write_text(
        """
universes:
  - universe_id: NASDAQ
    name: NASDAQ
    members:
      - {symbol: AAPL, exchange: SMART, primary_exchange: NASDAQ, currency: USD}
runs:
  - run_id: live-run
    universe: NASDAQ
    strategy: SESSION_HARD
    environment: LIVE
    risk: {risk_per_trade: 0.001}
    session:
      start: "09:30"
      end: "16:00"
      timezone: America/New_York
      calendar: XNYS
""",
        encoding="utf-8",
    )
    ibkr_path = tmp_path / "ibkr.yaml"
    ibkr_path.write_text(
        """
LIVE:
  environment: LIVE
  host: 127.0.0.1
  port: 4001
  client_id: 22
  expected_account: U123456
""",
        encoding="utf-8",
    )

    runtime = build_runtime(
        runs_config_path=runs_path,
        ibkr_config_path=ibkr_path,
        database_path=tmp_path / "runtime.sqlite3",
    )

    assert [item.environment for item in runtime.status().execution_environments] == [
        Environment.LIVE
    ]
    assert "ibkr_paper" not in runtime.status().as_dict()


def test_production_composition_starts_safely_with_all_runs_disabled(tmp_path) -> None:
    runs_path = tmp_path / "runs.yaml"
    runs_path.write_text(
        """
universes:
  - universe_id: CUSTOM_DEPLOYMENT
    name: Deployment configuration
    members:
      - {symbol: AAPL, exchange: SMART, primary_exchange: NASDAQ, currency: USD}
runs:
  - run_id: deployment-paper
    enabled: false
    universe: CUSTOM_DEPLOYMENT
    strategy: SESSION_HARD
    environment: PAPER
    risk: {risk_per_trade: 0.001, max_concurrent_positions: 1}
    session:
      start: "09:30"
      end: "16:00"
      timezone: America/New_York
      calendar: XNYS
""",
        encoding="utf-8",
    )
    ibkr_path = tmp_path / "ibkr.yaml"
    ibkr_path.write_text(
        """
PAPER:
  environment: PAPER
  host: 127.0.0.1
  port: 4003
  client_id: 81
  expected_account: DU_CONFIGURE_ME
""",
        encoding="utf-8",
    )
    runtime = build_runtime(
        runs_config_path=runs_path,
        ibkr_config_path=ibkr_path,
        database_path=tmp_path / "runtime.sqlite3",
    )

    status = asyncio.run(runtime.start())

    assert status.application is ApplicationState.READY
    assert status.runs[0].state is RunRuntimeState.DISABLED
    assert status.execution_environments[0].connected is False
