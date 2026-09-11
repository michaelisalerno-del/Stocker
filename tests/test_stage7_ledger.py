import sqlite3
from datetime import UTC, datetime

import pytest

from stocker_core.runs import Environment
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.execution_models import (
    BrokerFill,
    BrokerOrderIds,
    BrokerOrderStatus,
    EntryOrderType,
    OrderAction,
    OrderLifecycle,
    OrderPlan,
    OrderRole,
)


def _plan() -> OrderPlan:
    return OrderPlan(
        order_plan_id="plan-1",
        run_id="paper-run",
        signal_id="signal-1",
        strategy_id="TEST_EXECUTION",
        strategy_version="TEST_EXECUTION_V1",
        con_id=265598,
        symbol="AAPL",
        side=OrderAction.SELL,
        quantity=100,
        entry_order_type=EntryOrderType.MARKET,
        entry_reference=100.0,
        stop_price=101.0,
        target_price=98.0,
        environment=Environment.PAPER,
        created_at=datetime(2026, 9, 2, 14, 31, tzinfo=UTC),
    )


def _fill(
    execution_id: str,
    order_id: int,
    quantity: float,
    price: float,
    *,
    side: OrderAction,
    minute: int,
    commission: float | None = 0.0,
) -> BrokerFill:
    return BrokerFill(
        execution_id=execution_id,
        order_id=order_id,
        account="DU123456",
        environment=Environment.PAPER,
        con_id=265598,
        symbol="AAPL",
        side=side,
        quantity=quantity,
        price=price,
        executed_at=datetime(2026, 9, 2, 14, minute, tzinfo=UTC),
        commission=commission,
    )


def _submitted_ledger(tmp_path: object) -> ExecutionLedger:
    ledger = ExecutionLedger(tmp_path / "execution.sqlite3")  # type: ignore[operator]
    assert ledger.reserve(_plan(), expected_account="DU123456") is True
    ledger.record_submission(
        "plan-1", BrokerOrderIds(parent=101, stop=102, target=103), actual_account="DU123456"
    )
    return ledger


def test_signal_reservation_is_transactional_and_persistent_across_restart(
    tmp_path: object,
) -> None:
    path = tmp_path / "execution.sqlite3"  # type: ignore[operator]
    first = ExecutionLedger(path)

    assert first.reserve(_plan(), expected_account="DU123456") is True
    assert first.reserve(_plan(), expected_account="DU123456") is False
    assert ExecutionLedger(path).reserve(_plan(), expected_account="DU123456") is False


def test_stage8_global_fill_identity_schema_is_migrated_without_data_loss(tmp_path) -> None:
    path = tmp_path / "execution.sqlite3"
    ledger = _submitted_ledger(tmp_path)
    assert ledger.record_fill(_fill("stage8-exec", 101, 10, 100, side=OrderAction.SELL, minute=32))
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            ALTER TABLE execution_fills RENAME TO execution_fills_stage9;
            CREATE TABLE execution_fills (
                execution_id TEXT PRIMARY KEY,
                environment TEXT NOT NULL,
                account TEXT NOT NULL,
                order_id INTEGER NOT NULL,
                order_plan_id TEXT NOT NULL,
                role TEXT NOT NULL,
                con_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                executed_at TEXT NOT NULL,
                commission REAL
            );
            INSERT INTO execution_fills SELECT * FROM execution_fills_stage9;
            DROP TABLE execution_fills_stage9;
            """
        )

    migrated = ExecutionLedger(path)

    with sqlite3.connect(path) as connection:
        columns = connection.execute("PRAGMA table_info(execution_fills)").fetchall()
        primary_key = tuple(row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5] > 0)
        rows = connection.execute(
            "SELECT execution_id, environment, account FROM execution_fills"
        ).fetchall()
    assert primary_key == ("environment", "account", "execution_id")
    assert rows == [("stage8-exec", "PAPER", "DU123456")]
    assert migrated.get("plan-1").filled_quantity == 10  # type: ignore[union-attr]


def test_submission_persists_all_protective_broker_order_ids(tmp_path: object) -> None:
    ledger = _submitted_ledger(tmp_path)
    record = ledger.get("plan-1")

    assert record is not None
    assert record.status is OrderLifecycle.SUBMITTED
    assert record.parent_order_id == 101
    assert record.stop_order_id == 102
    assert record.target_order_id == 103
    assert record.expected_account == "DU123456"
    assert record.actual_account == "DU123456"
    assert ledger.order_role(Environment.PAPER, "DU123456", 102) is OrderRole.STOP


def test_partial_fills_aggregate_to_total_quantity_and_weighted_average(tmp_path: object) -> None:
    ledger = _submitted_ledger(tmp_path)

    assert ledger.record_fill(_fill("exec-1", 101, 40, 100.0, side=OrderAction.SELL, minute=32))
    partial = ledger.get("plan-1")
    assert partial is not None
    assert partial.status is OrderLifecycle.PARTIALLY_FILLED
    assert partial.filled_quantity == 40
    assert partial.average_fill_price == 100.0

    assert ledger.record_fill(_fill("exec-2", 101, 60, 101.0, side=OrderAction.SELL, minute=33))
    filled = ledger.get("plan-1")
    assert filled is not None
    assert filled.status is OrderLifecycle.FILLED
    assert filled.filled_quantity == 100
    assert filled.average_fill_price == pytest.approx(100.6)
    assert ledger.positions(Environment.PAPER, "DU123456")[0].quantity == -100


def test_duplicate_fill_callback_is_not_counted_twice(tmp_path: object) -> None:
    ledger = _submitted_ledger(tmp_path)
    fill = _fill("exec-1", 101, 40, 100.0, side=OrderAction.SELL, minute=32)

    assert ledger.record_fill(fill) is True
    assert ledger.record_fill(fill) is False
    record = ledger.get("plan-1")
    assert record is not None
    assert record.filled_quantity == 40


def test_rejection_creates_no_local_position_and_does_not_raise(tmp_path: object) -> None:
    ledger = ExecutionLedger(tmp_path / "execution.sqlite3")  # type: ignore[operator]
    assert ledger.reserve(_plan(), expected_account="DU123456")

    ledger.record_rejection("plan-1", "broker rejected test order")

    record = ledger.get("plan-1")
    assert record is not None
    assert record.status is OrderLifecycle.REJECTED
    assert record.rejection_reason == "broker rejected test order"
    assert ledger.positions(Environment.PAPER, "DU123456") == ()


def test_protective_exit_closes_position_and_records_realized_pnl(tmp_path: object) -> None:
    ledger = _submitted_ledger(tmp_path)
    ledger.record_fill(
        _fill("entry", 101, 100, 100.0, side=OrderAction.SELL, minute=32, commission=1.0)
    )
    ledger.record_fill(
        _fill("target", 103, 100, 98.0, side=OrderAction.BUY, minute=40, commission=1.0)
    )

    record = ledger.get("plan-1")
    assert record is not None
    assert record.status is OrderLifecycle.CLOSED
    assert record.closed_quantity == 100
    assert record.average_exit_price == 98.0
    assert record.realized_pnl == 198.0
    assert record.closed_at == datetime(2026, 9, 2, 14, 40, tzinfo=UTC)
    assert ledger.positions(Environment.PAPER, "DU123456") == ()


def test_unknown_order_fill_is_rejected_for_reconciliation(tmp_path: object) -> None:
    ledger = _submitted_ledger(tmp_path)

    accepted = ledger.record_fill(_fill("unknown", 999, 1, 100.0, side=OrderAction.SELL, minute=32))

    assert accepted is False
    assert ledger.get("plan-1").filled_quantity == 0  # type: ignore[union-attr]


def test_entry_order_rejection_status_updates_ledger_without_position(tmp_path: object) -> None:
    ledger = _submitted_ledger(tmp_path)

    accepted = ledger.record_order_status(
        BrokerOrderStatus(
            order_id=101,
            order_plan_id="plan-1",
            account="DU123456",
            environment=Environment.PAPER,
            status=OrderLifecycle.REJECTED,
            filled_quantity=0.0,
            remaining_quantity=100.0,
            reason="broker precaution rejected",
        )
    )

    record = ledger.get("plan-1")
    assert accepted is True
    assert record is not None
    assert record.status is OrderLifecycle.REJECTED
    assert record.rejection_reason == "broker precaution rejected"
    assert ledger.positions(Environment.PAPER, "DU123456") == ()


def test_pending_capacity_is_account_scoped(tmp_path):
    from dataclasses import replace

    from stocker_execution.execution_ledger import AdmissionRejected

    ledger = ExecutionLedger(tmp_path / "pending.sqlite")
    assert ledger.reserve(_plan(), expected_account="DU123456", max_positions=1)
    other = replace(_plan(), order_plan_id="other", signal_id="other", con_id=999)
    with pytest.raises(AdmissionRejected, match="CAPACITY_REACHED"):
        ExecutionLedger(ledger.path).reserve(other, expected_account="DU123456", max_positions=1)


def test_concurrent_cross_run_reservations_share_one_slot(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from threading import Barrier

    from stocker_execution.execution_ledger import AdmissionRejected

    path = tmp_path / "concurrent.sqlite"
    ledgers = [ExecutionLedger(path), ExecutionLedger(path)]
    barrier = Barrier(2)

    def admit(index):
        plan = replace(
            _plan(),
            order_plan_id=f"p{index}",
            signal_id=f"s{index}",
            run_id=f"r{index}",
            con_id=index + 1,
        )
        barrier.wait()
        try:
            return ledgers[index].reserve(plan, expected_account="DU123456", max_positions=1)
        except AdmissionRejected:
            return False

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(admit, (0, 1))) == [False, True]


def test_same_instrument_and_partial_fill_remainder(tmp_path):
    from dataclasses import replace

    from stocker_execution.execution_ledger import AdmissionRejected
    from stocker_execution.execution_models import BrokerPosition

    ledger = _submitted_ledger(tmp_path)
    ledger.record_fill(_fill("partial", 101, 40, 100, side=OrderAction.SELL, minute=32))
    other = replace(_plan(), order_plan_id="second", signal_id="second", run_id="other")
    position = (BrokerPosition("DU123456", 265598, "AAPL", -40, 100),)
    with pytest.raises(AdmissionRejected, match="POSITION_ALREADY_OPEN"):
        ledger.reserve(other, expected_account="DU123456", positions=position, max_positions=2)
    # 4,000 filled + 6,000 pending leaves 5,000, hence 50 shares for a new stock.
    assert ledger.reserve(
        replace(other, con_id=999),
        expected_account="DU123456",
        positions=position,
        max_positions=2,
        max_gross_notional=15000,
        broker_gross_notional=4000,
    )
    record = ledger.get("second")
    assert record.intended_quantity == 50
    assert record.risk_derived_quantity == 100
    assert record.sizing_reason == "GROSS_NOTIONAL_LIMIT"


def test_partial_cancellation_releases_remainder_not_position(tmp_path):
    from dataclasses import replace

    from stocker_execution.execution_models import BrokerPosition

    ledger = _submitted_ledger(tmp_path)
    ledger.record_fill(_fill("partial", 101, 40, 100, side=OrderAction.SELL, minute=32))
    ledger.record_order_status(
        BrokerOrderStatus(
            101, "plan-1", "DU123456", Environment.PAPER, OrderLifecycle.CANCELLED, 40, 60
        )
    )
    other = replace(_plan(), order_plan_id="second", signal_id="second", con_id=999)
    assert ledger.reserve(
        other,
        expected_account="DU123456",
        positions=(BrokerPosition("DU123456", 265598, "AAPL", -40, 100),),
        max_positions=2,
        max_gross_notional=15000,
        broker_gross_notional=4000,
    )
    assert ledger.get("second").intended_quantity == 100
    assert ledger.get("plan-1").filled_quantity == 40


@pytest.mark.parametrize(
    "terminal,reported",
    [
        (OrderLifecycle.FILLED, 100),
        (OrderLifecycle.CANCELLED, 40),
        (OrderLifecycle.REJECTED, 40),
    ],
)
@pytest.mark.parametrize("legacy_schema", [False, True])
def test_terminal_status_before_execution_details_retains_exposure(
    tmp_path, terminal, reported, legacy_schema
):
    import sqlite3
    from dataclasses import replace

    from stocker_execution.execution_ledger import AdmissionRejected
    from stocker_execution.execution_models import BrokerPosition

    ledger = _submitted_ledger(tmp_path)
    assert ledger.record_order_status(
        BrokerOrderStatus(
            101,
            "plan-1",
            "DU123456",
            Environment.PAPER,
            terminal,
            reported,
            100 - reported,
        )
    )
    if legacy_schema:
        with sqlite3.connect(ledger.path) as connection:
            connection.execute(
                "ALTER TABLE execution_plans DROP COLUMN broker_reported_entry_filled"
            )
    ledger = ExecutionLedger(ledger.path)  # Restart before execution callback.
    assert len(ledger.active_records(Environment.PAPER, "DU123456")) == 1
    assert ledger.positions(Environment.PAPER, "DU123456") == ()  # Never invent a broker fill.
    second = replace(_plan(), order_plan_id="second", signal_id="second", con_id=999)
    with pytest.raises(AdmissionRejected, match="CAPACITY_REACHED"):
        ledger.reserve(second, expected_account="DU123456", max_positions=1)
    with pytest.raises(AdmissionRejected, match="PENDING_ENTRY_CAPACITY_UNVERIFIED"):
        ledger.reserve(second, expected_account="DU123456", require_settled_entries=True)
    # Reconciliation obtains the cumulative report; an older zero report must
    # never undo positive evidence while the execution callback is still missing.
    ledger.record_order_status(
        BrokerOrderStatus(
            101,
            "plan-1",
            "DU123456",
            Environment.PAPER,
            terminal,
            reported,
            100 - reported,
        )
    )
    ledger.record_order_status(
        BrokerOrderStatus(
            101,
            "plan-1",
            "DU123456",
            Environment.PAPER,
            terminal,
            0,
            100,
        )
    )
    assert len(ledger.active_records(Environment.PAPER, "DU123456")) == 1
    ledger.record_fill(_fill("late", 101, reported, 100, side=OrderAction.SELL, minute=32))
    assert ledger.reserve(
        second,
        expected_account="DU123456",
        max_positions=2,
        max_gross_notional=15000,
        positions=(BrokerPosition("DU123456", 265598, "AAPL", -reported, 100),),
        broker_gross_notional=reported * 100,
        require_settled_entries=True,
    )
    assert ledger.get("second").intended_quantity == (50 if reported == 100 else 100)


def test_upgrade_preserves_conclusive_local_pre_submission_rejection(tmp_path):
    import sqlite3

    ledger = ExecutionLedger(tmp_path / "old-rejection.sqlite")
    plan = _plan()
    ledger.reserve(plan, expected_account="DU123456")
    ledger.record_rejection(plan.order_plan_id, "credit preview rejected before submission")
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("ALTER TABLE execution_plans DROP COLUMN broker_reported_entry_filled")
    restored = ExecutionLedger(ledger.path)
    assert restored.has_signal(plan.signal_id)
    assert restored.active_records(Environment.PAPER, "DU123456") == ()
