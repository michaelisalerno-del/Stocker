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

_EXECUTION_FILLS_DDL = """
CREATE TABLE execution_fills (
    execution_id TEXT NOT NULL,
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
    commission REAL,
    PRIMARY KEY (environment, account, execution_id)
);
"""


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
    initial_risk_budget: float | None
    per_share_initial_risk: float | None
    rejection_reason: str | None
    diagnostic: bool
    entry_limit_price: float | None = None
    entry_expires_at: datetime | None = None
    timeout_order_id: int | None = None
    deadline: datetime | None = None
    market_id: str | None = None
    method_spec_hash: str | None = None
    method_stop_price: float | None = None
    method_target_price: float | None = None
    exit_reason: str | None = None
    commission_total: float | None = None
    commissions_complete: bool = False

    @property
    def fill_relative_risk(self) -> float | None:
        if self.average_fill_price is None or self.method_stop_price is None:
            return None
        direction = 1 if self.side is OrderAction.BUY else -1
        return direction * (self.average_fill_price - self.method_stop_price)

    @property
    def fill_relative_reward(self) -> float | None:
        if self.average_fill_price is None or self.method_target_price is None:
            return None
        direction = 1 if self.side is OrderAction.BUY else -1
        return direction * (self.method_target_price - self.average_fill_price)

    def execution_metrics(self) -> dict[str, float | str | bool | None]:
        """Report from persisted prices; fills never redefine the nominal risk unit."""
        direction = 1 if self.side is OrderAction.BUY else -1
        risk = self.per_share_initial_risk
        if risk is not None and risk <= 0:
            risk = None
        slippage = (
            direction * (self.average_fill_price - self.entry_reference)
            if self.average_fill_price is not None else None
        )
        return {
            "entry_reference": self.entry_reference,
            "actual_fill_price": self.average_fill_price,
            "actual_exit_price": self.average_exit_price,
            "method_stop_price": self.method_stop_price,
            "method_target_price": self.method_target_price,
            "method_deadline": self.deadline.isoformat() if self.deadline else None,
            "nominal_r": risk,
            "fill_relative_risk": self.fill_relative_risk,
            "fill_relative_reward": self.fill_relative_reward,
            "entry_slippage_bps": slippage / self.entry_reference * 10000
            if slippage is not None and self.entry_reference > 0 else None,
            "entry_slippage_r": slippage / risk
            if slippage is not None and risk is not None else None,
            "method_reference_r": (
                direction * (self.average_exit_price - self.entry_reference) / risk
                if self.average_exit_price is not None and risk is not None else None
            ),
            "realized_execution_r": self.realized_pnl / (self.filled_quantity * risk)
            if self.realized_pnl is not None and self.filled_quantity > 0
            and risk is not None else None,
            "exit_reason": self.exit_reason,
            "commission_total": self.commission_total,
            "commissions_complete": self.commissions_complete,
        }


@dataclass(frozen=True, slots=True)
class BrokerOrderRecord:
    order_plan_id: str
    environment: Environment
    account: str
    order_id: int
    role: OrderRole
    status: OrderLifecycle


@dataclass(frozen=True, slots=True)
class BrokerPositionSnapshot:
    environment: Environment
    account: str
    con_id: int
    symbol: str
    quantity: float
    average_price: float
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class BrokerOpenOrderSnapshot:
    environment: Environment
    account: str
    order_id: int
    order_plan_id: str
    con_id: int
    symbol: str
    role: OrderRole
    status: OrderLifecycle
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class BrokerPositionMark:
    environment: Environment
    account: str
    con_id: int
    mark: float
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class ClosedTradeSummary:
    trades: int
    wins: int
    losses: int
    total_pnl: float


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
                CREATE TABLE IF NOT EXISTS broker_position_snapshot (
                    environment TEXT NOT NULL,
                    account TEXT NOT NULL,
                    con_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    average_price REAL NOT NULL,
                    observed_at TEXT NOT NULL,
                    PRIMARY KEY (environment, account, con_id)
                );
                CREATE TABLE IF NOT EXISTS broker_open_order_snapshot (
                    environment TEXT NOT NULL,
                    account TEXT NOT NULL,
                    order_id INTEGER NOT NULL,
                    order_plan_id TEXT NOT NULL,
                    con_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    PRIMARY KEY (environment, account, order_id)
                );
                CREATE TABLE IF NOT EXISTS broker_position_marks (
                    environment TEXT NOT NULL,
                    account TEXT NOT NULL,
                    con_id INTEGER NOT NULL,
                    mark REAL NOT NULL,
                    observed_at TEXT NOT NULL,
                    PRIMARY KEY (environment, account, con_id)
                );
                """
            )
            connection.execute(
                _EXECUTION_FILLS_DDL.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1)
            )
            self._migrate_execution_fill_identity(connection)
            self._ensure_column(connection, "execution_plans", "initial_risk_budget", "REAL")
            self._ensure_column(connection, "execution_plans", "per_share_initial_risk", "REAL")
            self._ensure_column(connection, "execution_plans", "entry_limit_price", "REAL")
            self._ensure_column(connection, "execution_plans", "entry_expires_at", "TEXT")
            for column in ("deadline", "market_id", "method_spec_hash"):
                self._ensure_column(connection, "execution_plans", column, "TEXT")
            self._ensure_column(connection, "execution_plans", "timeout_order_id", "INTEGER")
            for column in ("method_stop_price", "method_target_price", "commission_total"):
                self._ensure_column(connection, "execution_plans", column, "REAL")
            self._ensure_column(connection, "execution_plans", "exit_reason", "TEXT")
            self._ensure_column(
                connection, "execution_plans", "commissions_complete", "INTEGER NOT NULL DEFAULT 0"
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection, table: str, column: str, declaration: str
    ) -> None:
        columns = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @staticmethod
    def _migrate_execution_fill_identity(connection: sqlite3.Connection) -> None:
        """Upgrade Stage 7's global execution ID to an account-scoped identity."""

        columns = connection.execute("PRAGMA table_info(execution_fills)").fetchall()
        primary_key = tuple(
            str(row["name"])
            for row in sorted(columns, key=lambda row: int(row["pk"]))
            if int(row["pk"]) > 0
        )
        if primary_key == ("environment", "account", "execution_id"):
            return
        connection.execute("ALTER TABLE execution_fills RENAME TO execution_fills_stage8")
        connection.execute(_EXECUTION_FILLS_DDL)
        connection.execute(
            """
            INSERT INTO execution_fills (
                execution_id, environment, account, order_id, order_plan_id, role,
                con_id, symbol, side, quantity, price, executed_at, commission
            )
            SELECT execution_id, environment, account, order_id, order_plan_id, role,
                   con_id, symbol, side, quantity, price, executed_at, commission
            FROM execution_fills_stage8
            """
        )
        connection.execute("DROP TABLE execution_fills_stage8")

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
                        status, created_at, initial_risk_budget, per_share_initial_risk,
                        diagnostic, entry_limit_price, entry_expires_at, deadline,
                        market_id, method_spec_hash, method_stop_price, method_target_price
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
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
                        plan.initial_risk_budget,
                        plan.per_share_initial_risk,
                        int(plan.diagnostic),
                        plan.entry_limit_price,
                        plan.entry_expires_at.isoformat() if plan.entry_expires_at else None,
                        plan.deadline.isoformat() if plan.deadline else None,
                        plan.market_id,
                        plan.method_spec_hash,
                        plan.method_stop_price,
                        plan.method_target_price,
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
                    target_order_id = ?, timeout_order_id = ?, status = ?, submitted_at = ?
                WHERE order_plan_id = ?
                """,
                (
                    actual_account,
                    order_ids.parent,
                    order_ids.stop,
                    order_ids.target,
                    order_ids.timeout,
                    OrderLifecycle.SUBMITTED.value,
                    submitted_at,
                    order_plan_id,
                ),
            )
            for role, order_id in (
                (OrderRole.ENTRY, order_ids.parent),
                (OrderRole.STOP, order_ids.stop),
                (OrderRole.TARGET, order_ids.target),
                (OrderRole.TIMEOUT, order_ids.timeout),
            ):
                if order_id is None:
                    continue
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
                OrderRole.TIMEOUT: "timeout_order_id",
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
        """Record once, allowing later commission reports to enrich that execution."""

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
                if fill.commission is not None:
                    enriched = connection.execute(
                        """
                        UPDATE execution_fills SET commission = ?
                        WHERE environment = ? AND account = ? AND execution_id = ?
                          AND order_plan_id = ? AND commission IS NOT ?
                        """,
                        (fill.commission, fill.environment.value, fill.account,
                         fill.execution_id, plan_id, fill.commission),
                    )
                    if enriched.rowcount:
                        self._refresh_aggregate(connection, plan_id)
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

    def entry_fill_signal_ids(self) -> tuple[str, ...]:
        """Read actual entry fills in one query, without scanning skipped opportunities."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT signal_id FROM execution_plans WHERE filled_quantity > 0"
            ).fetchall()
        return tuple(str(row["signal_id"]) for row in rows)

    def attempted_signal_ids(self, run_id: str) -> frozenset[str]:
        """A persisted outcome or reserved plan is not a fresh admission opportunity."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT signal_id FROM execution_attempts WHERE run_id = ? "
                "UNION SELECT signal_id FROM execution_plans WHERE run_id = ?",
                (run_id, run_id),
            ).fetchall()
        return frozenset(str(row["signal_id"]) for row in rows)

    def has_signal(self, signal_id: str) -> bool:
        """Whether this signal already reserved a plan."""
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

    def list_records(
        self,
        *,
        run_id: str | None = None,
        run_ids: frozenset[str] | None = None,
        environment: Environment | None = None,
        symbol: str | None = None,
        statuses: frozenset[OrderLifecycle] | None = None,
        closed_only: bool = False,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[tuple[ExecutionRecord, ...], int]:
        """Page execution lineage for operational read models."""

        if not 1 <= limit <= 500:
            raise ValueError("execution record limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("execution record offset cannot be negative")
        clauses: list[str] = ["diagnostic = 0"]
        values: list[object] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if run_ids is not None:
            if run_ids:
                placeholders = ", ".join("?" for _ in run_ids)
                clauses.append(f"run_id IN ({placeholders})")
                values.extend(sorted(run_ids))
            else:
                clauses.append("1 = 0")
        if environment is not None:
            clauses.append("environment = ?")
            values.append(environment.value)
        if symbol is not None:
            clauses.append("symbol = ?")
            values.append(symbol.upper())
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            values.extend(status.value for status in sorted(statuses, key=lambda item: item.value))
        if closed_only:
            clauses.append("status = ?")
            values.append(OrderLifecycle.CLOSED.value)
        if start is not None:
            clauses.append("COALESCE(closed_at, submitted_at, created_at) >= ?")
            values.append(start.isoformat(timespec="microseconds"))
        if end is not None:
            clauses.append("COALESCE(closed_at, submitted_at, created_at) <= ?")
            values.append(end.isoformat(timespec="microseconds"))
        where = f"WHERE {' AND '.join(clauses)}"
        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM execution_plans {where}", values
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM execution_plans {where}
                ORDER BY COALESCE(closed_at, submitted_at, created_at) DESC, order_plan_id
                LIMIT ? OFFSET ?
                """,
                (*values, limit, offset),
            ).fetchall()
        return tuple(_record_from_row(row) for row in rows), total

    def closed_trade_summary(
        self,
        *,
        run_ids: frozenset[str] | None = None,
        environment: Environment | None = None,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> ClosedTradeSummary:
        """Aggregate the complete filtered closed ledger, independent of page size."""

        clauses = ["diagnostic = 0", "status = ?"]
        values: list[object] = [OrderLifecycle.CLOSED.value]
        if run_ids is not None:
            if run_ids:
                placeholders = ", ".join("?" for _ in run_ids)
                clauses.append(f"run_id IN ({placeholders})")
                values.extend(sorted(run_ids))
            else:
                clauses.append("1 = 0")
        if environment is not None:
            clauses.append("environment = ?")
            values.append(environment.value)
        if symbol is not None:
            clauses.append("symbol = ?")
            values.append(symbol.upper())
        if start is not None:
            clauses.append("COALESCE(closed_at, submitted_at, created_at) >= ?")
            values.append(start.isoformat(timespec="microseconds"))
        if end is not None:
            clauses.append("COALESCE(closed_at, submitted_at, created_at) <= ?")
            values.append(end.isoformat(timespec="microseconds"))
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS trades,
                       SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) AS losses,
                       COALESCE(SUM(realized_pnl), 0) AS total_pnl
                FROM execution_plans WHERE {" AND ".join(clauses)}
                """,
                values,
            ).fetchone()
        return ClosedTradeSummary(
            trades=int(row["trades"]),
            wins=int(row["wins"] or 0),
            losses=int(row["losses"] or 0),
            total_pnl=float(row["total_pnl"]),
        )

    def broker_orders(self, order_plan_id: str) -> tuple[BrokerOrderRecord, ...]:
        """Return normalized child-order state in meaningful bracket order."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT order_plan_id, environment, account, order_id, role, status
                FROM execution_broker_orders WHERE order_plan_id = ?
                ORDER BY CASE role WHEN 'ENTRY' THEN 0 WHEN 'STOP' THEN 1 ELSE 2 END
                """,
                (order_plan_id,),
            ).fetchall()
        return tuple(
            BrokerOrderRecord(
                order_plan_id=str(row["order_plan_id"]),
                environment=Environment(str(row["environment"])),
                account=str(row["account"]),
                order_id=int(row["order_id"]),
                role=OrderRole(str(row["role"])),
                status=OrderLifecycle(str(row["status"])),
            )
            for row in rows
        )

    def replace_broker_snapshot(
        self,
        *,
        environment: Environment,
        account: str,
        positions: tuple[BrokerPosition, ...],
        open_orders: tuple[BrokerOpenOrder, ...],
        observed_at: datetime,
    ) -> None:
        """Atomically retain the latest normalized IBKR exposure for dashboard reads."""

        timestamp = observed_at.isoformat(timespec="microseconds")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM broker_position_snapshot WHERE environment = ? AND account = ?",
                (environment.value, account),
            )
            connection.execute(
                "DELETE FROM broker_open_order_snapshot WHERE environment = ? AND account = ?",
                (environment.value, account),
            )
            connection.executemany(
                """
                INSERT INTO broker_position_snapshot
                (environment, account, con_id, symbol, quantity, average_price, observed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        environment.value,
                        account,
                        item.con_id,
                        item.symbol,
                        item.quantity,
                        item.average_price,
                        timestamp,
                    )
                    for item in positions
                    if item.account == account and item.quantity != 0.0
                ),
            )
            connection.executemany(
                """
                INSERT INTO broker_open_order_snapshot
                (environment, account, order_id, order_plan_id, con_id, symbol, role, status,
                 observed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        environment.value,
                        account,
                        item.order_id,
                        item.order_plan_id,
                        item.con_id,
                        item.symbol,
                        item.role.value,
                        item.status.value,
                        timestamp,
                    )
                    for item in open_orders
                    if item.environment is environment and item.account == account
                ),
            )

    def broker_position_snapshots(self) -> tuple[BrokerPositionSnapshot, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM broker_position_snapshot
                ORDER BY environment, account, symbol, con_id
                """
            ).fetchall()
        return tuple(
            BrokerPositionSnapshot(
                environment=Environment(str(row["environment"])),
                account=str(row["account"]),
                con_id=int(row["con_id"]),
                symbol=str(row["symbol"]),
                quantity=float(row["quantity"]),
                average_price=float(row["average_price"]),
                observed_at=datetime.fromisoformat(str(row["observed_at"])),
            )
            for row in rows
        )

    def record_position_mark(
        self,
        *,
        environment: Environment,
        account: str,
        con_id: int,
        mark: float,
        observed_at: datetime,
    ) -> None:
        """Retain the latest broker-authoritative mark for one reconciled position."""

        if con_id <= 0 or mark <= 0:
            raise ValueError("Position mark requires positive con_id and price")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO broker_position_marks
                (environment, account, con_id, mark, observed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(environment, account, con_id) DO UPDATE SET
                    mark = excluded.mark,
                    observed_at = excluded.observed_at
                """,
                (
                    environment.value,
                    account,
                    con_id,
                    mark,
                    observed_at.isoformat(timespec="microseconds"),
                ),
            )

    def broker_position_marks(self) -> tuple[BrokerPositionMark, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM broker_position_marks
                ORDER BY environment, account, con_id
                """
            ).fetchall()
        return tuple(
            BrokerPositionMark(
                environment=Environment(str(row["environment"])),
                account=str(row["account"]),
                con_id=int(row["con_id"]),
                mark=float(row["mark"]),
                observed_at=datetime.fromisoformat(str(row["observed_at"])),
            )
            for row in rows
        )

    def broker_open_order_snapshots(self) -> tuple[BrokerOpenOrderSnapshot, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM broker_open_order_snapshot
                ORDER BY environment, account, order_id
                """
            ).fetchall()
        return tuple(
            BrokerOpenOrderSnapshot(
                environment=Environment(str(row["environment"])),
                account=str(row["account"]),
                order_id=int(row["order_id"]),
                order_plan_id=str(row["order_plan_id"]),
                con_id=int(row["con_id"]),
                symbol=str(row["symbol"]),
                role=OrderRole(str(row["role"])),
                status=OrderLifecycle(str(row["status"])),
                observed_at=datetime.fromisoformat(str(row["observed_at"])),
            )
            for row in rows
        )

    def positions(self, environment: Environment, account: str) -> tuple[BrokerPosition, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT con_id, symbol,
                       SUM((CASE side WHEN 'BUY' THEN 1 ELSE -1 END)
                           * (filled_quantity - closed_quantity)) AS open_quantity,
                       SUM(filled_quantity * average_fill_price) AS entry_value,
                       SUM(filled_quantity) AS entry_quantity
                FROM execution_plans
                WHERE environment = ? AND actual_account = ?
                GROUP BY con_id, symbol
                HAVING ABS(open_quantity) > 0.000000001
                """,
                (environment.value, account),
            ).fetchall()
        return tuple(
            BrokerPosition(
                account=account,
                con_id=int(row["con_id"]),
                symbol=str(row["symbol"]),
                quantity=float(row["open_quantity"]),
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

    def record_for_order(
        self, environment: Environment, account: str, order_id: int
    ) -> ExecutionRecord | None:
        """Resolve any persisted leg, including deadline exits and closed trades."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT p.* FROM execution_plans p JOIN execution_broker_orders o
                ON p.order_plan_id = o.order_plan_id
                WHERE o.environment = ? AND o.account = ? AND o.order_id = ?
                """,
                (environment.value, account, order_id),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

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
        reported_commissions = [
            float(row["commission"]) for row in fills if row["commission"] is not None
        ]
        commission_total = sum(reported_commissions) if reported_commissions else None
        commissions_complete = bool(fills) and len(reported_commissions) == len(fills)
        exit_reason = None
        if closed:
            exit_reason = (
                "METHOD_DEADLINE" if exits[-1]["role"] == OrderRole.TIMEOUT.value
                else str(exits[-1]["role"])
            )
        if closed and entry_average is not None and exit_average is not None:
            closed_quantity = min(entry_quantity, exit_quantity)
            direction = 1 if plan["side"] == OrderAction.BUY.value else -1
            gross = direction * (exit_average - entry_average) * closed_quantity
            realized_pnl = gross - (commission_total or 0.0)
        connection.execute(
            """
            UPDATE execution_plans
            SET status = ?, filled_quantity = ?, average_fill_price = ?,
                closed_quantity = ?, average_exit_price = ?, opened_at = ?,
                closed_at = ?, realized_pnl = ?, exit_reason = ?, commission_total = ?,
                commissions_complete = ?
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
                exit_reason,
                commission_total,
                int(commissions_complete),
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
        timeout_order_id=_optional_int(row["timeout_order_id"]),
        deadline=_optional_datetime(row["deadline"]),
        market_id=row["market_id"],
        method_spec_hash=row["method_spec_hash"],
        method_stop_price=_optional_float(row["method_stop_price"]),
        method_target_price=_optional_float(row["method_target_price"]),
        exit_reason=row["exit_reason"],
        commission_total=_optional_float(row["commission_total"]),
        commissions_complete=bool(row["commissions_complete"]),
        filled_quantity=float(row["filled_quantity"]),
        average_fill_price=_optional_float(row["average_fill_price"]),
        closed_quantity=float(row["closed_quantity"]),
        average_exit_price=_optional_float(row["average_exit_price"]),
        submitted_at=_optional_datetime(row["submitted_at"]),
        opened_at=_optional_datetime(row["opened_at"]),
        closed_at=_optional_datetime(row["closed_at"]),
        realized_pnl=_optional_float(row["realized_pnl"]),
        initial_risk_budget=_optional_float(row["initial_risk_budget"]),
        per_share_initial_risk=_optional_float(row["per_share_initial_risk"]),
        entry_limit_price=_optional_float(row["entry_limit_price"]),
        entry_expires_at=(
            datetime.fromisoformat(row["entry_expires_at"]) if row["entry_expires_at"] else None
        ),
        rejection_reason=(
            str(row["rejection_reason"]) if row["rejection_reason"] is not None else None
        ),
        diagnostic=bool(row["diagnostic"]),
    )
