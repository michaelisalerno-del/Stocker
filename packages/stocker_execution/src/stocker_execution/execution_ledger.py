"""Persistent Stage 7 order, fill, position, and trade lineage."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from stocker_core.runs import Environment
from stocker_execution.execution_models import (
    BrokerFill,
    BrokerOpenOrder,
    BrokerOrderIds,
    BrokerOrderStatus,
    BrokerPosition,
    OrderAction,
    OrderLifecycle,
    OrderPlan,
    OrderRole,
)


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    order_plan_id: str
    signal_id: str
    run_id: str
    strategy_id: str
    strategy_version: str
    environment: Environment
    expected_account: str
    actual_account: str | None
    con_id: int
    symbol: str
    side: OrderAction
    intended_quantity: int
    entry_reference: float
    stop_price: float
    target_price: float
    status: OrderLifecycle
    parent_order_id: int | None
    stop_order_id: int | None
    target_order_id: int | None
    filled_quantity: float
    average_fill_price: float | None
    closed_quantity: float
    average_exit_price: float | None
    submitted_at: datetime | None
    opened_at: datetime | None
    closed_at: datetime | None
    realized_pnl: float | None
    rejection_reason: str | None
    diagnostic: bool


class ExecutionLedger:
    """SQLite execution store with an atomic signal idempotency reservation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS execution_plans (
                    order_plan_id TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    expected_account TEXT NOT NULL,
                    actual_account TEXT,
                    con_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    intended_quantity INTEGER NOT NULL,
                    entry_reference REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    status TEXT NOT NULL,
                    parent_order_id INTEGER,
                    stop_order_id INTEGER,
                    target_order_id INTEGER,
                    created_at TEXT NOT NULL,
                    submitted_at TEXT,
                    opened_at TEXT,
                    closed_at TEXT,
                    filled_quantity REAL NOT NULL DEFAULT 0,
                    average_fill_price REAL,
                    closed_quantity REAL NOT NULL DEFAULT 0,
                    average_exit_price REAL,
                    realized_pnl REAL,
                    rejection_reason TEXT,
                    diagnostic INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS execution_broker_orders (
                    environment TEXT NOT NULL,
                    account TEXT NOT NULL,
                    order_id INTEGER NOT NULL,
                    order_plan_id TEXT NOT NULL REFERENCES execution_plans(order_plan_id),
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY (environment, account, order_id),
                    UNIQUE (order_plan_id, role)
                );
                CREATE TABLE IF NOT EXISTS execution_fills (
                    execution_id TEXT PRIMARY KEY,
                    environment TEXT NOT NULL,
                    account TEXT NOT NULL,
                    order_id INTEGER NOT NULL,
                    order_plan_id TEXT NOT NULL REFERENCES execution_plans(order_plan_id),
                    role TEXT NOT NULL,
                    con_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    executed_at TEXT NOT NULL,
                    commission REAL
                );
                CREATE TABLE IF NOT EXISTS execution_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    signal_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    expected_account TEXT NOT NULL,
                    actual_account TEXT,
                    result_code TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    attempted_at TEXT NOT NULL
                );
                """
            )

    def reserve(self, plan: OrderPlan, *, expected_account: str) -> bool:
        """Atomically reserve a Stage 6 signal before any broker transmission."""

        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO execution_plans (
                        order_plan_id, signal_id, run_id, strategy_id, strategy_version,
                        environment, expected_account, con_id, symbol, side,
                        intended_quantity, entry_reference, stop_price, target_price,
                        status, created_at, diagnostic
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plan.order_plan_id,
                        plan.signal_id,
                        plan.run_id,
                        plan.strategy_id,
                        plan.strategy_version,
                        plan.environment.value,
                        expected_account,
                        plan.con_id,
                        plan.symbol,
                        plan.side.value,
                        plan.quantity,
                        plan.entry_reference,
                        plan.stop_price,
                        plan.target_price,
                        OrderLifecycle.PLANNED.value,
                        plan.created_at.isoformat(timespec="microseconds"),
                        int(plan.diagnostic),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def mark_submitting(self, order_plan_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE execution_plans SET status = ? WHERE order_plan_id = ?",
                (OrderLifecycle.SUBMITTING.value, order_plan_id),
            )

    def record_submission(
        self, order_plan_id: str, order_ids: BrokerOrderIds, *, actual_account: str
    ) -> None:
        """Persist the coherent entry/stop/target broker identity transactionally."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT environment FROM execution_plans WHERE order_plan_id = ?",
                (order_plan_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown order plan: {order_plan_id}")
            environment = str(row["environment"])
            submitted_at = datetime.now().astimezone().isoformat(timespec="microseconds")
            connection.execute(
                """
                UPDATE execution_plans
                SET actual_account = ?, parent_order_id = ?, stop_order_id = ?,
                    target_order_id = ?, status = ?, submitted_at = ?
                WHERE order_plan_id = ?
                """,
                (
                    actual_account,
                    order_ids.parent,
                    order_ids.stop,
                    order_ids.target,
                    OrderLifecycle.SUBMITTED.value,
                    submitted_at,
                    order_plan_id,
                ),
            )
            for role, order_id in (
                (OrderRole.ENTRY, order_ids.parent),
                (OrderRole.STOP, order_ids.stop),
                (OrderRole.TARGET, order_ids.target),
            ):
                connection.execute(
                    """
                    INSERT INTO execution_broker_orders (
                        environment, account, order_id, order_plan_id, role, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        environment,
                        actual_account,
                        order_id,
                        order_plan_id,
                        role.value,
                        OrderLifecycle.SUBMITTED.value,
                    ),
                )

    def record_rejection(self, order_plan_id: str, reason: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE execution_plans
                SET status = ?, rejection_reason = ?
                WHERE order_plan_id = ?
                """,
                (OrderLifecycle.REJECTED.value, reason, order_plan_id),
            )

    def record_attempt(
        self,
        *,
        run_id: str,
        signal_id: str,
        environment: Environment,
        expected_account: str,
        actual_account: str | None,
        result_code: str,
        detail: str,
        attempted_at: datetime,
    ) -> None:
        """Persist every execution outcome, including pre-plan safety blocks."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO execution_attempts (
                    run_id, signal_id, environment, expected_account, actual_account,
                    result_code, detail, attempted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    signal_id,
                    environment.value,
                    expected_account,
                    actual_account,
                    result_code,
                    detail,
                    attempted_at.isoformat(timespec="microseconds"),
                ),
            )

    def recover_open_order(self, order: BrokerOpenOrder) -> bool:
        """Bind an open IBKR order to its reserved deterministic order reference."""

        with self._connect() as connection:
            plan = connection.execute(
                """
                SELECT environment, expected_account, status
                FROM execution_plans WHERE order_plan_id = ?
                """,
                (order.order_plan_id,),
            ).fetchone()
            if (
                plan is None
                or str(plan["environment"]) != order.environment.value
                or str(plan["expected_account"]) != order.account
                or str(plan["status"])
                in {
                    OrderLifecycle.CLOSED.value,
                    OrderLifecycle.CANCELLED.value,
                    OrderLifecycle.REJECTED.value,
                }
            ):
                return False
            column = {
                OrderRole.ENTRY: "parent_order_id",
                OrderRole.STOP: "stop_order_id",
                OrderRole.TARGET: "target_order_id",
            }[order.role]
            try:
                connection.execute(
                    """
                    INSERT INTO execution_broker_orders (
                        environment, account, order_id, order_plan_id, role, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order.environment.value,
                        order.account,
                        order.order_id,
                        order.order_plan_id,
                        order.role.value,
                        order.status.value,
                    ),
                )
                connection.execute(
                    f"""
                    UPDATE execution_plans
                    SET {column} = ?, actual_account = ?, status = ?,
                        submitted_at = COALESCE(submitted_at, ?)
                    WHERE order_plan_id = ?
                    """,
                    (
                        order.order_id,
                        order.account,
                        OrderLifecycle.SUBMITTED.value,
                        datetime.now().astimezone().isoformat(timespec="microseconds"),
                        order.order_plan_id,
                    ),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def recover_entry_fill_order(self, fill: BrokerFill) -> bool:
        """Recover a filled parent from its IBKR order reference after a crash."""

        if not fill.order_plan_id:
            return False
        with self._connect() as connection:
            plan = connection.execute(
                """
                SELECT environment, expected_account, side, status
                FROM execution_plans WHERE order_plan_id = ?
                """,
                (fill.order_plan_id,),
            ).fetchone()
            if (
                plan is None
                or str(plan["environment"]) != fill.environment.value
                or str(plan["expected_account"]) != fill.account
                or str(plan["side"]) != fill.side.value
                or str(plan["status"])
                in {
                    OrderLifecycle.CLOSED.value,
                    OrderLifecycle.CANCELLED.value,
                    OrderLifecycle.REJECTED.value,
                }
            ):
                return False
            try:
                connection.execute(
                    """
                    INSERT INTO execution_broker_orders (
                        environment, account, order_id, order_plan_id, role, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fill.environment.value,
                        fill.account,
                        fill.order_id,
                        fill.order_plan_id,
                        OrderRole.ENTRY.value,
                        OrderLifecycle.FILLED.value,
                    ),
                )
                connection.execute(
                    """
                    UPDATE execution_plans
                    SET parent_order_id = ?, actual_account = ?, status = ?,
                        submitted_at = COALESCE(submitted_at, ?)
                    WHERE order_plan_id = ?
                    """,
                    (
                        fill.order_id,
                        fill.account,
                        OrderLifecycle.SUBMITTED.value,
                        fill.executed_at.isoformat(timespec="microseconds"),
                        fill.order_plan_id,
                    ),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def order_role(self, environment: Environment, account: str, order_id: int) -> OrderRole | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT role FROM execution_broker_orders
                WHERE environment = ? AND account = ? AND order_id = ?
                """,
                (environment.value, account, order_id),
            ).fetchone()
        return OrderRole(str(row["role"])) if row is not None else None

    def record_fill(self, fill: BrokerFill) -> bool:
        """Record one execution once and derive its trade/position aggregate."""

        if fill.quantity <= 0.0 or fill.price <= 0.0:
            return False
        with self._connect() as connection:
            order = connection.execute(
                """
                SELECT order_plan_id, role FROM execution_broker_orders
                WHERE environment = ? AND account = ? AND order_id = ?
                """,
                (fill.environment.value, fill.account, fill.order_id),
            ).fetchone()
            if order is None:
                return False
            plan_id = str(order["order_plan_id"])
            role = str(order["role"])
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO execution_fills (
                    execution_id, environment, account, order_id, order_plan_id, role,
                    con_id, symbol, side, quantity, price, executed_at, commission
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.execution_id,
                    fill.environment.value,
                    fill.account,
                    fill.order_id,
                    plan_id,
                    role,
                    fill.con_id,
                    fill.symbol,
                    fill.side.value,
                    fill.quantity,
                    fill.price,
                    fill.executed_at.isoformat(timespec="microseconds"),
                    fill.commission,
                ),
            )
            if cursor.rowcount == 0:
                return False
            self._refresh_aggregate(connection, plan_id)
        return True

    def record_order_status(self, status: BrokerOrderStatus) -> bool:
        """Persist one normalized status without treating submission as exposure."""

        with self._connect() as connection:
            order = connection.execute(
                """
                SELECT order_plan_id, role FROM execution_broker_orders
                WHERE environment = ? AND account = ? AND order_id = ?
                """,
                (status.environment.value, status.account, status.order_id),
            ).fetchone()
            if order is None:
                return False
            plan_id = str(order["order_plan_id"])
            connection.execute(
                """
                UPDATE execution_broker_orders SET status = ?
                WHERE environment = ? AND account = ? AND order_id = ?
                """,
                (
                    status.status.value,
                    status.environment.value,
                    status.account,
                    status.order_id,
                ),
            )
            if (
                str(order["role"]) == OrderRole.ENTRY.value
                and status.filled_quantity == 0.0
                and status.status in {OrderLifecycle.REJECTED, OrderLifecycle.CANCELLED}
            ):
                connection.execute(
                    """
                    UPDATE execution_plans SET status = ?, rejection_reason = ?
                    WHERE order_plan_id = ? AND filled_quantity = 0
                    """,
                    (status.status.value, status.reason or None, plan_id),
                )
        return True

    def get(self, order_plan_id: str) -> ExecutionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM execution_plans WHERE order_plan_id = ?",
                (order_plan_id,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def has_signal(self, signal_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM execution_plans WHERE signal_id = ?", (signal_id,)
            ).fetchone()
        return row is not None

    def active_records(self, environment: Environment, account: str) -> tuple[ExecutionRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM execution_plans
                WHERE environment = ? AND expected_account = ?
                  AND status NOT IN (?, ?, ?)
                ORDER BY created_at, order_plan_id
                """,
                (
                    environment.value,
                    account,
                    OrderLifecycle.CLOSED.value,
                    OrderLifecycle.CANCELLED.value,
                    OrderLifecycle.REJECTED.value,
                ),
            ).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    def positions(self, environment: Environment, account: str) -> tuple[BrokerPosition, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT con_id, symbol,
                       SUM(filled_quantity - closed_quantity) AS open_quantity,
                       SUM(filled_quantity * average_fill_price) AS entry_value,
                       SUM(filled_quantity) AS entry_quantity
                FROM execution_plans
                WHERE environment = ? AND actual_account = ? AND side = ?
                GROUP BY con_id, symbol
                HAVING ABS(open_quantity) > 0.000000001
                """,
                (environment.value, account, OrderAction.SELL.value),
            ).fetchall()
        return tuple(
            BrokerPosition(
                account=account,
                con_id=int(row["con_id"]),
                symbol=str(row["symbol"]),
                quantity=-float(row["open_quantity"]),
                average_price=float(row["entry_value"]) / float(row["entry_quantity"]),
            )
            for row in rows
        )

    def known_order_ids(self, environment: Environment, account: str) -> frozenset[int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT order_id FROM execution_broker_orders
                WHERE environment = ? AND account = ?
                """,
                (environment.value, account),
            ).fetchall()
        return frozenset(int(row["order_id"]) for row in rows)

    def _refresh_aggregate(self, connection: sqlite3.Connection, plan_id: str) -> None:
        fills = connection.execute(
            """
            SELECT role, quantity, price, executed_at, commission
            FROM execution_fills WHERE order_plan_id = ?
            ORDER BY executed_at, execution_id
            """,
            (plan_id,),
        ).fetchall()
        plan = connection.execute(
            "SELECT intended_quantity, side FROM execution_plans WHERE order_plan_id = ?",
            (plan_id,),
        ).fetchone()
        if plan is None:
            raise ValueError(f"unknown order plan: {plan_id}")
        entries = [row for row in fills if row["role"] == OrderRole.ENTRY.value]
        exits = [row for row in fills if row["role"] != OrderRole.ENTRY.value]
        entry_quantity = sum(float(row["quantity"]) for row in entries)
        exit_quantity = sum(float(row["quantity"]) for row in exits)
        entry_average = _weighted_average(entries, entry_quantity)
        exit_average = _weighted_average(exits, exit_quantity)
        opened_at = min((str(row["executed_at"]) for row in entries), default=None)
        closed = entry_quantity > 0.0 and exit_quantity >= entry_quantity
        closed_at = (
            max((str(row["executed_at"]) for row in exits), default=None) if closed else None
        )
        if closed:
            status = OrderLifecycle.CLOSED
        elif entry_quantity >= int(plan["intended_quantity"]):
            status = OrderLifecycle.FILLED
        elif entry_quantity > 0.0:
            status = OrderLifecycle.PARTIALLY_FILLED
        else:
            status = OrderLifecycle.SUBMITTED
        realized_pnl = None
        if closed and entry_average is not None and exit_average is not None:
            closed_quantity = min(entry_quantity, exit_quantity)
            gross = (entry_average - exit_average) * closed_quantity
            commissions = sum(float(row["commission"] or 0.0) for row in fills)
            realized_pnl = gross - commissions
        connection.execute(
            """
            UPDATE execution_plans
            SET status = ?, filled_quantity = ?, average_fill_price = ?,
                closed_quantity = ?, average_exit_price = ?, opened_at = ?,
                closed_at = ?, realized_pnl = ?
            WHERE order_plan_id = ?
            """,
            (
                status.value,
                entry_quantity,
                entry_average,
                exit_quantity,
                exit_average,
                opened_at,
                closed_at,
                realized_pnl,
                plan_id,
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _weighted_average(rows: list[sqlite3.Row], quantity: float) -> float | None:
    if quantity <= 0.0:
        return None
    return sum(float(row["quantity"]) * float(row["price"]) for row in rows) / quantity


def _optional_datetime(value: object) -> datetime | None:
    return datetime.fromisoformat(str(value)) if value is not None else None


def _optional_float(value: object) -> float | None:
    return float(str(value)) if value is not None else None


def _optional_int(value: object) -> int | None:
    return int(str(value)) if value is not None else None


def _record_from_row(row: sqlite3.Row) -> ExecutionRecord:
    return ExecutionRecord(
        order_plan_id=str(row["order_plan_id"]),
        signal_id=str(row["signal_id"]),
        run_id=str(row["run_id"]),
        strategy_id=str(row["strategy_id"]),
        strategy_version=str(row["strategy_version"]),
        environment=Environment(str(row["environment"])),
        expected_account=str(row["expected_account"]),
        actual_account=str(row["actual_account"]) if row["actual_account"] is not None else None,
        con_id=int(row["con_id"]),
        symbol=str(row["symbol"]),
        side=OrderAction(str(row["side"])),
        intended_quantity=int(row["intended_quantity"]),
        entry_reference=float(row["entry_reference"]),
        stop_price=float(row["stop_price"]),
        target_price=float(row["target_price"]),
        status=OrderLifecycle(str(row["status"])),
        parent_order_id=_optional_int(row["parent_order_id"]),
        stop_order_id=_optional_int(row["stop_order_id"]),
        target_order_id=_optional_int(row["target_order_id"]),
        filled_quantity=float(row["filled_quantity"]),
        average_fill_price=_optional_float(row["average_fill_price"]),
        closed_quantity=float(row["closed_quantity"]),
        average_exit_price=_optional_float(row["average_exit_price"]),
        submitted_at=_optional_datetime(row["submitted_at"]),
        opened_at=_optional_datetime(row["opened_at"]),
        closed_at=_optional_datetime(row["closed_at"]),
        realized_pnl=_optional_float(row["realized_pnl"]),
        rejection_reason=(
            str(row["rejection_reason"]) if row["rejection_reason"] is not None else None
        ),
        diagnostic=bool(row["diagnostic"]),
    )
