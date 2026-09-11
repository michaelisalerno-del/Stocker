"""Frozen downstream semantics; every broker interaction terminates at a fake."""

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from stocker_core.runs import Environment, RunConfig
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerFill,
    BrokerOpenOrder,
    BrokerOrderIds,
    BrokerPosition,
    OrderAction,
    OrderLifecycle,
    OrderRole,
)
from stocker_execution.runtime import RuntimeStore
from stocker_execution.session_hard_method import SessionHardMethod, TradeEvent
from stocker_execution.session_hard_structure_d import SignalStatus
from stocker_execution.stage7 import Stage7ExecutionService, build_order_plan
from test_method_package import candidate
from test_stage7_order_plan import _decision


def armed():
    _, admitted = candidate()
    assert admitted.q1_eligible
    admitted = replace(
        admitted,
        p0=100.0,
        m_price=1.0,
        up_trigger=100.20,
        down_trigger=99.80,
    )
    method = SessionHardMethod(clock=lambda: admitted.t0)
    method.restore_signals((admitted,))
    return method, admitted


def triggered(long=True, seconds=180):
    method, signal = armed()
    level = signal.up_trigger if long else signal.down_trigger
    signal = method.observe_trades(
        {signal.underlying_con_id: (TradeEvent(signal.t0 + timedelta(seconds=seconds), level, 1),)}
    )[0]
    return method, signal


def submitted(tmp_path, signal, tick=0.01):
    plan = build_order_plan(
        order_intent=signal,
        risk_decision=_decision(signal),
        environment=Environment.PAPER,
        minimum_tick=tick,
        created_at=signal.entry_timestamp,
    )
    ledger = ExecutionLedger(tmp_path / "execution.sqlite")
    ledger.reserve(plan, expected_account="DU_TEST")
    ledger.record_submission(
        plan.order_plan_id, BrokerOrderIds(101, 102, 103, 104), actual_account="DU_TEST"
    )
    return plan, ledger


def fill(plan, order, price, timestamp, commission=None):
    entry = order == 101
    return BrokerFill(
        execution_id=f"fill-{order}",
        order_id=order,
        account="DU_TEST",
        environment=Environment.PAPER,
        con_id=plan.con_id,
        symbol=plan.symbol,
        side=plan.side
        if entry
        else OrderAction.SELL
        if plan.side is OrderAction.BUY
        else OrderAction.BUY,
        quantity=plan.quantity,
        price=price,
        executed_at=timestamp,
        commission=commission,
    )


@pytest.mark.parametrize("long", [True, False])
def test_exact_geometry_and_late_first_break_deadline(long):
    method, signal = triggered(long)
    direction = 1 if long else -1
    assert signal.entry_reference == 100 + direction * 0.20
    assert signal.stop_price == signal.entry_reference - direction * 0.50
    assert signal.target_price == signal.entry_reference + direction * 1.00
    assert signal.deadline == signal.t0 + timedelta(minutes=15)
    assert signal.deadline - signal.entry_timestamp == timedelta(minutes=12)
    other = signal.down_trigger if long else signal.up_trigger
    assert (
        method.observe_trades(
            {
                signal.underlying_con_id: (
                    TradeEvent(signal.entry_timestamp + timedelta(seconds=1), other, 2),
                )
            }
        )
        == ()
    )
    assert method.signals[0] == signal
    with pytest.raises(ValueError, match="ordered TRADES"):
        method.observe_entry_bars({})


@pytest.mark.parametrize("seconds,enters", [(0, True), (299.999, True), (300, False), (301, False)])
def test_first_break_window_is_half_open(seconds, enters):
    _, signal = triggered(seconds=seconds)
    assert (signal.status is SignalStatus.ENTRY_TRIGGERED) == enters
    if not enters:
        assert signal.reason == "ENTRY_WINDOW_EXPIRED"


def test_no_print_candidate_expires_at_exact_window_end():
    method, signal = armed()
    assert (
        method.expire_waiting_before(signal.t0 + timedelta(minutes=5) - timedelta(microseconds=1))
        == ()
    )
    expired = method.expire_waiting_before(signal.t0 + timedelta(minutes=5))
    assert len(expired) == 1
    assert expired[0].reason == "ENTRY_WINDOW_EXPIRED"


@pytest.mark.parametrize("long", [True, False])
@pytest.mark.parametrize("slippage", [-0.03, 0.03])
def test_fill_independent_bracket_and_nominal_r_persist(tmp_path, long, slippage):
    _, signal = triggered(long)
    plan, ledger = submitted(tmp_path, signal)
    direction = 1 if long else -1
    price = signal.entry_reference + direction * slippage
    entry = fill(plan, 101, price, signal.entry_timestamp, commission=1.0)
    assert ledger.record_fill(entry)
    record = ExecutionLedger(ledger.path).get(plan.order_plan_id)
    assert record.entry_reference == signal.entry_reference
    assert record.average_fill_price == price
    assert record.stop_price == signal.stop_price
    assert record.target_price == signal.target_price
    assert record.method_stop_price == signal.stop_price
    assert record.method_target_price == signal.target_price
    assert record.fill_relative_risk == pytest.approx(0.50 + slippage)
    assert record.fill_relative_reward == pytest.approx(1.00 - slippage)
    assert record.per_share_initial_risk == 0.50
    assert record.deadline == signal.t0 + timedelta(minutes=15)
    ledger.record_fill(
        fill(plan, 103, signal.target_price, signal.deadline - timedelta(seconds=1), 1.0)
    )
    record = ledger.get(plan.order_plan_id)
    assert record.exit_reason == "TARGET"
    metrics = record.execution_metrics()
    assert metrics["method_reference_r"] == 2.0
    assert metrics["realized_execution_r"] == pytest.approx(
        ((1 - slippage) * plan.quantity - 2) / (0.5 * plan.quantity)
    )
    assert metrics["entry_slippage_r"] == pytest.approx(slippage / 0.5)


@pytest.mark.parametrize("order,reason", [(102, "STOP"), (103, "TARGET"), (104, "METHOD_DEADLINE")])
def test_exit_reason_and_delayed_commissions_survive_reload(tmp_path, order, reason):
    _, signal = triggered()
    plan, ledger = submitted(tmp_path, signal)
    entry = fill(plan, 101, signal.entry_reference, signal.entry_timestamp)
    exit_fill = fill(plan, order, 100.0, signal.deadline)
    ledger.record_fill(entry)
    ledger.record_fill(exit_fill)
    before = ledger.get(plan.order_plan_id)
    assert before.exit_reason == reason
    assert before.commission_total is None
    # A later IBKR commission report enriches the same execution, never a second fill.
    assert ledger.record_fill(replace(entry, commission=0.75)) is False
    assert ledger.record_fill(replace(exit_fill, commission=0.80)) is False
    after = ExecutionLedger(ledger.path).get(plan.order_plan_id)
    assert after.realized_pnl == pytest.approx(before.realized_pnl - 1.55)
    assert after.commission_total == 1.55 and after.commissions_complete
    assert after.filled_quantity == after.closed_quantity == plan.quantity
    assert after.exit_reason == reason and after.status is OrderLifecycle.CLOSED


@pytest.mark.parametrize("known_break", [False, True])
def test_restart_restores_frozen_candidate_without_recalculation(tmp_path, known_break):
    method, signal = triggered() if known_break else armed()
    store = RuntimeStore(tmp_path / "runtime.sqlite")
    store.save_signals((signal,), signal.t0)
    restarted = SessionHardMethod(clock=lambda: signal.t0 + timedelta(minutes=4))
    restarted.restore_signals(RuntimeStore(store.path).load_signals(signal.run_id))
    assert restarted.signals == method.signals
    if known_break:
        assert (
            restarted.observe_trades(
                {
                    signal.underlying_con_id: (
                        TradeEvent(signal.t0 + timedelta(minutes=4), signal.down_trigger, 2),
                    )
                }
            )
            == ()
        )
    else:
        assert (
            restarted.expire_waiting_before(signal.t0 + timedelta(minutes=5))[0].reason
            == "ENTRY_WINDOW_EXPIRED"
        )


def test_broker_deadline_survives_gtc_brackets_and_three_minute_late_entry(tmp_path):
    from stocker_execution.ibkr import IbkrConnection
    from test_stage7_ibkr import FakeOrderClient, _config, _instrument

    _, signal = triggered()
    now = datetime.now(UTC)
    signal = replace(
        signal,
        t0=now - timedelta(minutes=3),
        entry_timestamp=now,
        deadline=now + timedelta(minutes=12),
    )
    plan, _ = submitted(tmp_path, signal)
    plan = replace(plan, con_id=_instrument().con_id)
    client = FakeOrderClient()
    broker = IbkrConnection(_config(), client=client, execution_enabled=True)
    asyncio.run(broker.connect())
    asyncio.run(broker.submit_protected_order(plan, _instrument()))
    _, target, stop, close = [order for _, order in client.placed]
    assert target.tif == stop.tif == "GTC"
    assert close.orderType == "MKT" and close.action == "SELL"
    assert close.conditions[0].time == signal.deadline.strftime("%Y%m%d %H:%M:%S UTC")
    assert close.conditions[0].isMore
    assert all(o.ocaType == 2 and o.ocaGroup == close.ocaGroup for o in (stop, target))
    broker.disconnect()
    asyncio.run(broker.connect())
    assert len(client.placed) == 4  # Recovery does not replace/re-time the broker-owned close.


def test_tick_grid_prices_remain_distinct_from_frozen_method_geometry(tmp_path):
    _, signal = triggered()
    signal = replace(signal, stop_price=99.697, target_price=101.206)
    plan, ledger = submitted(tmp_path, signal)
    assert plan.stop_price == 99.70 and plan.target_price == 101.20
    assert plan.method_stop_price == 99.697 and plan.method_target_price == 101.206
    ledger.record_fill(fill(plan, 101, 100.23, signal.entry_timestamp))
    record = ExecutionLedger(ledger.path).get(plan.order_plan_id)
    assert record.method_stop_price == 99.697 and record.method_target_price == 101.206
    assert record.stop_price == 99.70 and record.target_price == 101.20


def test_execution_diagnostics_migration_preserves_existing_rows(tmp_path):
    from test_stage7_ledger import _submitted_ledger

    ledger = _submitted_ledger(tmp_path)
    columns_added = {
        "method_stop_price",
        "method_target_price",
        "exit_reason",
        "commission_total",
        "commissions_complete",
    }
    with sqlite3.connect(ledger.path) as db:
        for column in columns_added:
            db.execute(f"ALTER TABLE execution_plans DROP COLUMN {column}")
        before = db.execute("SELECT * FROM execution_plans").fetchall()
        old_columns = [row[1] for row in db.execute("PRAGMA table_info(execution_plans)")]
    migrated = ExecutionLedger(ledger.path)
    with sqlite3.connect(migrated.path) as db:
        assert (
            db.execute(f"SELECT {', '.join(old_columns)} FROM execution_plans").fetchall() == before
        )
        assert db.execute(
            "SELECT method_stop_price, exit_reason, commission_total, commissions_complete "
            "FROM execution_plans"
        ).fetchone() == (None, None, None, 0)


@pytest.mark.parametrize("state", ["pending", "open", "deadline_closed", "missing_deadline"])
def test_reconciliation_recovers_current_method_without_replacing_deadline(tmp_path, state):
    from test_stage7_execution import FakeExecutionBroker

    _, signal = triggered()
    plan, ledger = submitted(tmp_path, signal)
    pending = state in {"pending", "missing_deadline"}
    if state == "missing_deadline":
        # Crash before broker IDs were committed; only three legs come back from IBKR.
        ledger = ExecutionLedger(tmp_path / "crashed.sqlite")
        assert ledger.reserve(plan, expected_account="DU_TEST")
    entries = () if pending else (fill(plan, 101, 100.23, signal.entry_timestamp),)
    exits = (fill(plan, 104, 100.40, signal.deadline),) if state == "deadline_closed" else ()
    roles = (
        (101, OrderRole.ENTRY),
        (102, OrderRole.STOP),
        (103, OrderRole.TARGET),
        (104, OrderRole.TIMEOUT),
    )
    orders = tuple(
        BrokerOpenOrder(
            order_id,
            plan.order_plan_id,
            "DU_TEST",
            Environment.PAPER,
            plan.con_id,
            plan.symbol,
            role,
            OrderLifecycle.SUBMITTED,
        )
        for order_id, role in roles
        if state != "deadline_closed"
        and (pending or role is not OrderRole.ENTRY)
        and not (state == "missing_deadline" and role is OrderRole.TIMEOUT)
    )
    positions = (
        (BrokerPosition("DU_TEST", plan.con_id, plan.symbol, plan.quantity, 100.23),)
        if state == "open"
        else ()
    )
    broker = FakeExecutionBroker(
        account="DU_TEST", open_orders=orders, fills=entries + exits, positions=positions
    )
    restarted = Stage7ExecutionService(
        run=RunConfig(
            run_id=signal.run_id,
            universe=signal.universe_id,
            strategy=signal.strategy_id,
            environment=Environment.PAPER,
        ),
        expected_account="DU_TEST",
        broker=broker,
        ledger=ExecutionLedger(ledger.path),
        clock=lambda: signal.deadline,
    )
    result = asyncio.run(restarted.reconcile())
    if state == "missing_deadline":
        assert not result.ok and "missing method deadline order" in result.detail
        assert broker.submitted == []
        return
    assert result.ok
    record = ExecutionLedger(ledger.path).get(plan.order_plan_id)
    assert record.deadline == signal.t0 + timedelta(minutes=15)
    assert record.entry_reference == signal.entry_reference
    assert record.method_stop_price == signal.stop_price
    assert record.method_target_price == signal.target_price
    assert record.method_spec_hash == signal.method_spec_hash
    assert record.strategy_version == signal.strategy_version
    assert record.timeout_order_id == 104
    assert record.average_fill_price == (None if state == "pending" else 100.23)
    assert broker.submitted == []
    if state == "deadline_closed":
        assert record.status is OrderLifecycle.CLOSED and record.exit_reason == "METHOD_DEADLINE"


def test_dashboard_reports_reference_fill_nominal_r_and_deadline_exit(tmp_path):
    from test_stage10_dashboard import _seed_authoritative_state

    service = _seed_authoritative_state(tmp_path)
    _, signal = triggered()
    signal = replace(signal, run_id=service.config.runs[0].run_id)
    plan, ledger = submitted(tmp_path, signal)
    service.ledger = ledger
    ledger.record_fill(fill(plan, 101, 100.23, signal.entry_timestamp, 1.0))
    ledger.record_fill(fill(plan, 104, 100.40, signal.deadline, 1.0))
    order = service.order_detail(plan.order_plan_id)
    assert order["execution"]["entry_reference"] == 100.20
    assert order["execution"]["actual_fill_price"] == 100.23
    assert order["execution"]["method_stop_price"] == 99.70
    trades = service.trades(environment=None, start=None, end=None)["items"]
    assert trades[0]["exit_reason"] == "METHOD_DEADLINE"
    assert trades[0]["r"] == pytest.approx(
        ((100.40 - 100.23) * plan.quantity - 2) / (plan.quantity * 0.50)
    )
    assert trades[0]["method_spec_hash"] == signal.method_spec_hash
