import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

from stocker_core.runs import Environment, RunConfig, RunRiskConfig
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerAccountState,
    BrokerFill,
    BrokerOpenOrder,
    BrokerOrderIds,
    BrokerOrderStatus,
    BrokerPosition,
    EntryOrderType,
    OrderAction,
    OrderLifecycle,
    OrderPlan,
    OrderRole,
)
from stocker_execution.ibkr import QualifiedInstrument
from stocker_execution.session_hard_structure_d import (
    EntryBar,
    SessionHardStructureDStrategy,
    SignalStatus,
    StrategySignal,
)
from stocker_execution.stage7 import (
    ExecutionResultCode,
    Stage7ExecutionService,
    Stage7PaperRuntime,
)


class FakeExecutionBroker:
    def __init__(
        self,
        *,
        environment: Environment = Environment.PAPER,
        account: str = "DU123456",
        connected: bool = True,
        positions: tuple[BrokerPosition, ...] = (),
        open_orders: tuple[BrokerOpenOrder, ...] = (),
        fills: tuple[BrokerFill, ...] = (),
        statuses: tuple[BrokerOrderStatus, ...] = (),
        reject: bool = False,
        minimum_tick_error: Exception | None = None,
        account_state_error: Exception | None = None,
    ) -> None:
        self.environment = environment
        self.account = account
        self.is_connected = connected
        self.connection_epoch = 1
        self._positions = positions
        self._open_orders = open_orders
        self._fills = fills
        self._statuses = statuses
        self.reject = reject
        self.minimum_tick_error = minimum_tick_error
        self.account_state_error = account_state_error
        self.submitted = []
        self.account_reads = 0

    async def account_state(self) -> BrokerAccountState:
        self.account_reads += 1
        if self.account_state_error is not None:
            raise self.account_state_error
        return BrokerAccountState(
            self.environment,
            self.account,
            100_000.0,
            200_000.0,
            self.is_connected,
            self._positions,
        )

    async def minimum_tick(self, instrument: QualifiedInstrument) -> float:
        if self.minimum_tick_error is not None:
            raise self.minimum_tick_error
        return 0.01

    async def submit_protected_order(self, plan: object, instrument: object) -> BrokerOrderIds:
        if self.reject:
            raise RuntimeError("simulated broker rejection")
        self.submitted.append((plan, instrument))
        return BrokerOrderIds(parent=101, stop=102, target=103)

    async def read_open_orders(self) -> tuple[BrokerOpenOrder, ...]:
        return self._open_orders

    async def read_fills(self) -> tuple[BrokerFill, ...]:
        return self._fills

    async def read_positions(self) -> tuple[BrokerPosition, ...]:
        return self._positions

    async def read_order_statuses(self) -> tuple[BrokerOrderStatus, ...]:
        return self._statuses


class TriggeredStage6Strategy(SessionHardStructureDStrategy):
    def observe_entry_bars(self, bars_by_con_id: object) -> tuple[StrategySignal, ...]:
        return (_intent(),)


def _run(environment: Environment = Environment.PAPER) -> RunConfig:
    return RunConfig(
        run_id="paper-run",
        universe="NASDAQ",
        strategy="SESSION_HARD",
        environment=environment,
        risk=RunRiskConfig(risk_per_trade=0.001, max_concurrent_positions=5),
    )


def _intent() -> StrategySignal:
    timestamp = datetime(2026, 9, 2, 14, 31, tzinfo=UTC)
    return StrategySignal(
        strategy_id="SESSION_HARD_HIGH_PRE_MOVE_DOWN_STRUCTURE_D",
        strategy_version="SESSION_HARD_STRUCTURE_D_V1",
        signal_id="signal-1",
        run_id="paper-run",
        underlying_con_id=265598,
        symbol="AAPL",
        universe_id="NASDAQ",
        session=date(2026, 9, 2),
        t0=datetime(2026, 9, 2, 14, 30, tzinfo=UTC),
        status=SignalStatus.ENTRY_TRIGGERED,
        reason="STRUCTURE_D_DOWN_FIRST_TOUCH",
        pre_move_m=0.8,
        cohort_percentile=75.0,
        band=None,
        session_hard_score=0.9999,
        session_hard_checkpoint=6,
        session_hard_qualified=True,
        feature_calculation_version="STAGE5_PRE_MOVE_V1",
        side="SHORT",
        direction="DOWN",
        candidate_rank=1,
        selected=True,
        p0=100.4,
        m_price=2.0,
        entry_level=100.0,
        entry_reference=100.0,
        entry_timestamp=timestamp,
        signal_timestamp=timestamp,
    )


def _instrument() -> QualifiedInstrument:
    return QualifiedInstrument("AAPL", 265598, "SMART", "NASDAQ", "USD", "STK")


def _service(
    path: Path,
    broker: FakeExecutionBroker,
    *,
    run: RunConfig | None = None,
    expected_account: str = "DU123456",
) -> Stage7ExecutionService:
    return Stage7ExecutionService(
        run=run or _run(),
        expected_account=expected_account,
        broker=broker,
        ledger=ExecutionLedger(path),
        clock=lambda: datetime(2026, 9, 2, 14, 31, 1, tzinfo=UTC),
    )


def test_paper_run_reconciles_then_submits_one_protected_plan(tmp_path: Path) -> None:
    broker = FakeExecutionBroker()
    service = _service(tmp_path / "ledger.sqlite3", broker)

    assert asyncio.run(service.reconcile()).ok is True
    result = asyncio.run(service.execute(_intent(), _instrument()))

    assert result.code is ExecutionResultCode.SUBMITTED
    assert len(broker.submitted) == 1
    plan = broker.submitted[0][0]
    assert plan.side is OrderAction.SELL
    assert plan.quantity == 100
    assert plan.stop_price == 101.0
    assert plan.target_price == 98.0
    assert result.order_ids == BrokerOrderIds(101, 102, 103)


def test_live_run_requires_reconciliation_before_transmission(tmp_path: Path) -> None:
    broker = FakeExecutionBroker(environment=Environment.LIVE, account="U123456")
    service = _service(
        tmp_path / "ledger.sqlite3",
        broker,
        run=_run(Environment.LIVE),
        expected_account="U123456",
    )

    result = asyncio.run(service.execute(_intent(), _instrument()))

    assert result.code is ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED
    assert result.actual_account == "U123456"
    assert broker.account_reads == 1
    assert broker.submitted == []


def test_wrong_account_blocks_submission(tmp_path: Path) -> None:
    broker = FakeExecutionBroker(account="DU999999")
    service = _service(tmp_path / "ledger.sqlite3", broker)

    result = asyncio.run(service.execute(_intent(), _instrument()))

    assert result.code is ExecutionResultCode.ACCOUNT_OR_ENVIRONMENT_MISMATCH
    assert result.run_id == "paper-run"
    assert result.environment is Environment.PAPER
    assert result.expected_account == "DU123456"
    assert result.actual_account == "DU999999"
    assert broker.submitted == []


def test_disconnected_broker_blocks_submission(tmp_path: Path) -> None:
    broker = FakeExecutionBroker(connected=False)
    service = _service(tmp_path / "ledger.sqlite3", broker)

    result = asyncio.run(service.execute(_intent(), _instrument()))

    assert result.code is ExecutionResultCode.BROKER_DISCONNECTED
    assert broker.submitted == []


def test_new_orders_require_reconciliation_and_reconnect_invalidates_it(tmp_path: Path) -> None:
    broker = FakeExecutionBroker()
    service = _service(tmp_path / "ledger.sqlite3", broker)

    before = asyncio.run(service.execute(_intent(), _instrument()))
    assert before.code is ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED

    assert asyncio.run(service.reconcile()).ok
    broker.connection_epoch += 1
    after_reconnect = asyncio.run(service.execute(_intent(), _instrument()))
    assert after_reconnect.code is ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED
    assert broker.submitted == []


def test_stage7_does_not_add_an_unowned_expiry_to_stage6_intent(tmp_path: Path) -> None:
    broker = FakeExecutionBroker()
    service = _service(tmp_path / "ledger.sqlite3", broker)
    assert asyncio.run(service.reconcile()).ok
    stale_time = datetime(2026, 9, 2, 14, 20, tzinfo=UTC)
    stale = replace(_intent(), entry_timestamp=stale_time, signal_timestamp=stale_time)

    result = asyncio.run(service.execute(stale, _instrument()))

    assert result.code is ExecutionResultCode.SUBMITTED
    assert len(broker.submitted) == 1


def test_account_state_failure_is_candidate_local(tmp_path: Path) -> None:
    broker = FakeExecutionBroker(account_state_error=RuntimeError("request interrupted"))
    service = _service(tmp_path / "ledger.sqlite3", broker)

    result = asyncio.run(service.execute(_intent(), _instrument()))

    assert result.code is ExecutionResultCode.ACCOUNT_STATE_UNAVAILABLE
    assert "request interrupted" in result.detail
    assert broker.submitted == []


def test_account_state_failure_blocks_reconciliation_without_raising(tmp_path: Path) -> None:
    broker = FakeExecutionBroker(account_state_error=RuntimeError("request interrupted"))

    result = asyncio.run(_service(tmp_path / "ledger.sqlite3", broker).reconcile())

    assert result.ok is False
    assert result.code is ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED
    assert "account state unavailable" in result.detail


def test_minimum_tick_failure_rejects_only_that_candidate(tmp_path: Path) -> None:
    broker = FakeExecutionBroker(minimum_tick_error=RuntimeError("no contract details"))
    service = _service(tmp_path / "ledger.sqlite3", broker)
    assert asyncio.run(service.reconcile()).ok

    result = asyncio.run(service.execute(_intent(), _instrument()))

    assert result.code is ExecutionResultCode.ORDER_PLAN_UNAVAILABLE
    assert "no contract details" in result.detail
    assert broker.submitted == []


def test_same_signal_cannot_submit_twice(tmp_path: Path) -> None:
    broker = FakeExecutionBroker()
    service = _service(tmp_path / "ledger.sqlite3", broker)
    assert asyncio.run(service.reconcile()).ok

    first = asyncio.run(service.execute(_intent(), _instrument()))
    second = asyncio.run(service.execute(_intent(), _instrument()))

    assert first.code is ExecutionResultCode.SUBMITTED
    assert second.code is ExecutionResultCode.DUPLICATE_ORDER_BLOCKED
    assert len(broker.submitted) == 1


def test_restart_replay_reconciles_known_open_orders_then_blocks_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    first_broker = FakeExecutionBroker()
    first_service = _service(path, first_broker)
    assert asyncio.run(first_service.reconcile()).ok
    assert (
        asyncio.run(first_service.execute(_intent(), _instrument())).code
        is ExecutionResultCode.SUBMITTED
    )

    orders = tuple(
        BrokerOpenOrder(
            order_id=order_id,
            order_plan_id="29bb461d6c532bfdb3429030ed14ddb0",
            account="DU123456",
            environment=Environment.PAPER,
            con_id=265598,
            symbol="AAPL",
            role=role,
            status=OrderLifecycle.SUBMITTED,
        )
        for order_id, role in (
            (101, OrderRole.ENTRY),
            (102, OrderRole.STOP),
            (103, OrderRole.TARGET),
        )
    )
    restarted_broker = FakeExecutionBroker(open_orders=orders)
    restarted = _service(path, restarted_broker)

    assert asyncio.run(restarted.reconcile()).ok
    result = asyncio.run(restarted.execute(_intent(), _instrument()))
    assert result.code is ExecutionResultCode.DUPLICATE_ORDER_BLOCKED
    assert restarted_broker.submitted == []


def test_reconnect_recovers_pending_bracket_from_deterministic_order_reference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    plan = OrderPlan(
        "recover-plan",
        "paper-run",
        "recover-signal",
        "strategy",
        "v1",
        265598,
        "AAPL",
        OrderAction.SELL,
        100,
        EntryOrderType.MARKET,
        100.0,
        101.0,
        98.0,
        Environment.PAPER,
        datetime(2026, 9, 2, 14, 31, tzinfo=UTC),
    )
    ledger = ExecutionLedger(path)
    assert ledger.reserve(plan, expected_account="DU123456")
    ledger.mark_submitting(plan.order_plan_id)
    orders = tuple(
        BrokerOpenOrder(
            order_id,
            plan.order_plan_id,
            "DU123456",
            Environment.PAPER,
            265598,
            "AAPL",
            role,
            OrderLifecycle.SUBMITTED,
        )
        for order_id, role in (
            (201, OrderRole.ENTRY),
            (202, OrderRole.STOP),
            (203, OrderRole.TARGET),
        )
    )

    reconciliation = asyncio.run(
        _service(path, FakeExecutionBroker(open_orders=orders)).reconcile()
    )

    assert reconciliation.ok is True
    assert ledger.known_order_ids(Environment.PAPER, "DU123456") == {201, 202, 203}


def test_reconnect_recovers_parent_that_filled_before_ids_were_persisted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    plan = OrderPlan(
        "recover-filled-plan",
        "paper-run",
        "recover-filled-signal",
        "strategy",
        "v1",
        265598,
        "AAPL",
        OrderAction.SELL,
        100,
        EntryOrderType.MARKET,
        100.0,
        101.0,
        98.0,
        Environment.PAPER,
        datetime(2026, 9, 2, 14, 31, tzinfo=UTC),
    )
    ledger = ExecutionLedger(path)
    assert ledger.reserve(plan, expected_account="DU123456")
    ledger.mark_submitting(plan.order_plan_id)
    children = tuple(
        BrokerOpenOrder(
            order_id,
            plan.order_plan_id,
            "DU123456",
            Environment.PAPER,
            265598,
            "AAPL",
            role,
            OrderLifecycle.SUBMITTED,
        )
        for order_id, role in ((202, OrderRole.STOP), (203, OrderRole.TARGET))
    )
    entry_fill = BrokerFill(
        "filled-before-persist",
        201,
        "DU123456",
        Environment.PAPER,
        265598,
        "AAPL",
        OrderAction.SELL,
        100,
        100.0,
        datetime(2026, 9, 2, 14, 31, 30, tzinfo=UTC),
        0.0,
        order_plan_id=plan.order_plan_id,
    )
    broker = FakeExecutionBroker(
        open_orders=children,
        fills=(entry_fill,),
        positions=(BrokerPosition("DU123456", 265598, "AAPL", -100, 100.0),),
    )

    reconciliation = asyncio.run(_service(path, broker).reconcile())

    record = ledger.get(plan.order_plan_id)
    assert reconciliation.ok is True
    assert record is not None
    assert record.parent_order_id == 201
    assert record.filled_quantity == 100


def test_broker_rejection_is_recorded_without_position(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    broker = FakeExecutionBroker(reject=True)
    service = _service(path, broker)
    assert asyncio.run(service.reconcile()).ok

    result = asyncio.run(service.execute(_intent(), _instrument()))

    assert result.code is ExecutionResultCode.BROKER_REJECTED
    record = ExecutionLedger(path).get(result.order_plan.order_plan_id)  # type: ignore[union-attr]
    assert record is not None
    assert record.status is OrderLifecycle.REJECTED
    assert ExecutionLedger(path).positions(Environment.PAPER, "DU123456") == ()


def test_reconnect_ingests_completed_entry_rejection_and_reconciles(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    first_broker = FakeExecutionBroker()
    first_service = _service(path, first_broker)
    assert asyncio.run(first_service.reconcile()).ok
    submitted = asyncio.run(first_service.execute(_intent(), _instrument()))
    assert submitted.code is ExecutionResultCode.SUBMITTED

    rejection = BrokerOrderStatus(
        101,
        submitted.order_plan.order_plan_id,  # type: ignore[union-attr]
        "DU123456",
        Environment.PAPER,
        OrderLifecycle.REJECTED,
        0.0,
        100.0,
        "broker rejected while process was offline",
    )
    restarted = _service(path, FakeExecutionBroker(statuses=(rejection,)))

    reconciliation = asyncio.run(restarted.reconcile())
    record = ExecutionLedger(path).get(submitted.order_plan.order_plan_id)  # type: ignore[union-attr]
    assert reconciliation.ok is True
    assert record is not None
    assert record.status is OrderLifecycle.REJECTED


def test_known_broker_and_local_position_reconciles(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    broker = FakeExecutionBroker()
    service = _service(path, broker)
    assert asyncio.run(service.reconcile()).ok
    result = asyncio.run(service.execute(_intent(), _instrument()))
    plan_id = result.order_plan.order_plan_id  # type: ignore[union-attr]
    ledger = ExecutionLedger(path)
    ledger.record_fill(
        BrokerFill(
            "exec-1",
            101,
            "DU123456",
            Environment.PAPER,
            265598,
            "AAPL",
            OrderAction.SELL,
            100,
            100.0,
            datetime(2026, 9, 2, 14, 32, tzinfo=UTC),
            0.0,
        )
    )
    broker._positions = (BrokerPosition("DU123456", 265598, "AAPL", -100, 100.0),)
    broker._open_orders = tuple(
        BrokerOpenOrder(
            order_id,
            plan_id,
            "DU123456",
            Environment.PAPER,
            265598,
            "AAPL",
            role,
            OrderLifecycle.SUBMITTED,
        )
        for order_id, role in ((102, OrderRole.STOP), (103, OrderRole.TARGET))
    )

    assert asyncio.run(_service(path, broker).reconcile()).ok is True


def test_known_position_without_both_protective_children_blocks(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    broker = FakeExecutionBroker()
    service = _service(path, broker)
    assert asyncio.run(service.reconcile()).ok
    assert (
        asyncio.run(service.execute(_intent(), _instrument())).code is ExecutionResultCode.SUBMITTED
    )
    ledger = ExecutionLedger(path)
    ledger.record_fill(
        BrokerFill(
            "entry",
            101,
            "DU123456",
            Environment.PAPER,
            265598,
            "AAPL",
            OrderAction.SELL,
            100,
            100.0,
            datetime(2026, 9, 2, 14, 32, tzinfo=UTC),
            0.0,
        )
    )
    broker._positions = (BrokerPosition("DU123456", 265598, "AAPL", -100, 100.0),)
    broker._open_orders = ()

    reconciliation = asyncio.run(_service(path, broker).reconcile())

    assert reconciliation.ok is False
    assert "missing protective" in reconciliation.detail


def test_unexpected_broker_position_blocks_new_orders(tmp_path: Path) -> None:
    broker = FakeExecutionBroker(positions=(BrokerPosition("DU123456", 999, "MSFT", 10, 250.0),))
    service = _service(tmp_path / "ledger.sqlite3", broker)

    reconciliation = asyncio.run(service.reconcile())

    assert reconciliation.ok is False
    assert reconciliation.code is ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED
    assert "unexpected broker position" in reconciliation.detail


def test_unexpected_open_broker_order_blocks_new_orders(tmp_path: Path) -> None:
    unknown = BrokerOpenOrder(
        999,
        "external-plan",
        "DU123456",
        Environment.PAPER,
        265598,
        "AAPL",
        OrderRole.ENTRY,
        OrderLifecycle.SUBMITTED,
    )
    broker = FakeExecutionBroker(open_orders=(unknown,))
    service = _service(tmp_path / "ledger.sqlite3", broker)

    reconciliation = asyncio.run(service.reconcile())

    assert reconciliation.ok is False
    assert "unexpected broker open order" in reconciliation.detail


def test_unrelated_terminal_order_history_does_not_block_reconciliation(tmp_path: Path) -> None:
    unrelated = BrokerOrderStatus(
        999,
        "manual-old-order",
        "DU123456",
        Environment.PAPER,
        OrderLifecycle.CANCELLED,
        0.0,
        0.0,
        "",
    )

    reconciliation = asyncio.run(
        _service(
            tmp_path / "ledger.sqlite3", FakeExecutionBroker(statuses=(unrelated,))
        ).reconcile()
    )

    assert reconciliation.ok is True


def test_refresh_rereconciles_new_unknown_exposure(tmp_path: Path) -> None:
    broker = FakeExecutionBroker()
    service = _service(tmp_path / "ledger.sqlite3", broker)
    assert asyncio.run(service.reconcile()).ok
    broker._positions = (BrokerPosition("DU123456", 999, "MSFT", 10, 250.0),)

    refreshed = asyncio.run(service.refresh_broker_state())

    assert refreshed.ok is False
    assert "unexpected broker position" in refreshed.detail


def test_normal_runtime_batch_consumes_selected_stage6_intents(tmp_path: Path) -> None:
    broker = FakeExecutionBroker()
    service = _service(tmp_path / "ledger.sqlite3", broker)
    assert asyncio.run(service.reconcile()).ok

    results = asyncio.run(service.execute_ready_intents((_intent(),), {265598: _instrument()}))

    assert [result.code for result in results] == [ExecutionResultCode.SUBMITTED]
    assert len(broker.submitted) == 1


def test_normal_runtime_observes_stage6_then_executes_its_intent(tmp_path: Path) -> None:
    broker = FakeExecutionBroker()
    service = _service(tmp_path / "ledger.sqlite3", broker)
    assert asyncio.run(service.reconcile()).ok
    runtime = Stage7PaperRuntime(strategy=TriggeredStage6Strategy(), execution=service)

    results = asyncio.run(
        runtime.observe_and_execute(
            {265598: (EntryBar(datetime(2026, 9, 2, 14, 31, tzinfo=UTC), 100, 101, 99),)},
            {265598: _instrument()},
        )
    )

    assert [result.code for result in results] == [ExecutionResultCode.SUBMITTED]
    assert len(broker.submitted) == 1


def test_missing_instrument_attempt_preserves_known_connected_account(tmp_path: Path) -> None:
    broker = FakeExecutionBroker()
    service = _service(tmp_path / "ledger.sqlite3", broker)
    assert asyncio.run(service.reconcile()).ok

    results = asyncio.run(service.execute_ready_intents((_intent(),), {}))

    assert results[0].code is ExecutionResultCode.ORDER_PLAN_UNAVAILABLE
    assert results[0].actual_account == "DU123456"
