"""Stage 7 risk, planning, and PAPER execution orchestration."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import StrEnum
from math import floor, isfinite
from typing import Protocol

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
    OrderPlan,
)
from stocker_execution.ibkr import QualifiedInstrument
from stocker_execution.session_hard_structure_d import SignalStatus, StrategySignal


class RiskRejection(StrEnum):
    INVALID_RISK_CONFIG = "INVALID_RISK_CONFIG"
    INVALID_STOP_DISTANCE = "INVALID_STOP_DISTANCE"
    ZERO_QUANTITY = "ZERO_QUANTITY"
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    CAPACITY_REACHED = "CAPACITY_REACHED"
    ACCOUNT_STATE_UNAVAILABLE = "ACCOUNT_STATE_UNAVAILABLE"


class ExecutionResultCode(StrEnum):
    SUBMITTED = "SUBMITTED"
    LIVE_EXECUTION_DISABLED = "LIVE_EXECUTION_DISABLED"
    ACCOUNT_OR_ENVIRONMENT_MISMATCH = "ACCOUNT_OR_ENVIRONMENT_MISMATCH"
    BROKER_DISCONNECTED = "BROKER_DISCONNECTED"
    EXECUTION_RECONCILIATION_REQUIRED = "EXECUTION_RECONCILIATION_REQUIRED"
    DUPLICATE_ORDER_BLOCKED = "DUPLICATE_ORDER_BLOCKED"
    BROKER_REJECTED = "BROKER_REJECTED"
    INVALID_RISK_CONFIG = "INVALID_RISK_CONFIG"
    INVALID_STOP_DISTANCE = "INVALID_STOP_DISTANCE"
    ZERO_QUANTITY = "ZERO_QUANTITY"
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    CAPACITY_REACHED = "CAPACITY_REACHED"
    ACCOUNT_STATE_UNAVAILABLE = "ACCOUNT_STATE_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class Stage7RiskDecision:
    approved: bool
    reason: str
    risk_budget: float = 0.0
    per_share_risk: float = 0.0
    quantity: int = 0
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0


@dataclass(frozen=True, slots=True)
class ExecutionAttempt:
    code: ExecutionResultCode
    detail: str
    order_plan: OrderPlan | None = None
    order_ids: BrokerOrderIds | None = None


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    ok: bool
    code: ExecutionResultCode
    detail: str


class ExecutionBroker(Protocol):
    @property
    def environment(self) -> Environment: ...

    @property
    def account(self) -> str: ...

    @property
    def is_connected(self) -> bool: ...

    @property
    def connection_epoch(self) -> int: ...

    async def account_state(self) -> BrokerAccountState: ...

    async def minimum_tick(self, instrument: QualifiedInstrument) -> float: ...

    async def submit_protected_order(
        self, plan: OrderPlan, instrument: QualifiedInstrument
    ) -> BrokerOrderIds: ...

    async def read_open_orders(self) -> tuple[BrokerOpenOrder, ...]: ...

    async def read_fills(self) -> tuple[BrokerFill, ...]: ...

    async def read_positions(self) -> tuple[BrokerPosition, ...]: ...

    async def read_order_statuses(self) -> tuple[BrokerOrderStatus, ...]: ...


class Stage7RiskEngine:
    """Size one Stage 6 intent without making broker calls."""

    def evaluate(
        self,
        *,
        order_intent: StrategySignal,
        account_state: BrokerAccountState,
        risk_config: RunRiskConfig | None,
    ) -> Stage7RiskDecision:
        if risk_config is None:
            return _rejected(RiskRejection.INVALID_RISK_CONFIG)
        risk_fraction = float(risk_config.risk_per_trade)
        capacity = risk_config.max_concurrent_positions
        if (
            not isfinite(risk_fraction)
            or risk_fraction <= 0.0
            or risk_fraction > 1.0
            or (capacity is not None and capacity <= 0)
        ):
            return _rejected(RiskRejection.INVALID_RISK_CONFIG)
        equity = account_state.equity
        if not account_state.connected or equity is None or not isfinite(equity) or equity <= 0.0:
            return _rejected(RiskRejection.ACCOUNT_STATE_UNAVAILABLE)
        if any(
            position.con_id == order_intent.underlying_con_id and position.quantity != 0.0
            for position in account_state.positions
        ):
            return _rejected(RiskRejection.POSITION_ALREADY_OPEN)
        if capacity is not None:
            open_positions = sum(position.quantity != 0.0 for position in account_state.positions)
            if open_positions >= capacity:
                return _rejected(RiskRejection.CAPACITY_REACHED)

        entry = order_intent.entry_reference
        m_price = order_intent.m_price
        if (
            order_intent.status is not SignalStatus.ENTRY_TRIGGERED
            or not order_intent.selected
            or order_intent.side != "SHORT"
            or entry is None
            or not isfinite(entry)
            or entry <= 0.0
            or m_price is None
            or not isfinite(m_price)
            or m_price <= 0.0
            or not isfinite(order_intent.stop_distance_m)
            or order_intent.stop_distance_m <= 0.0
            or not isfinite(order_intent.target_distance_m)
            or order_intent.target_distance_m <= 0.0
        ):
            return _rejected(RiskRejection.INVALID_STOP_DISTANCE)

        stop = entry + order_intent.stop_distance_m * m_price
        target = entry - order_intent.target_distance_m * m_price
        per_share_risk = abs(entry - stop)
        if (
            not isfinite(stop)
            or not isfinite(target)
            or stop <= entry
            or target <= 0.0
            or target >= entry
            or per_share_risk <= 0.0
        ):
            return _rejected(RiskRejection.INVALID_STOP_DISTANCE)

        risk_budget = equity * risk_fraction
        quantity = floor(risk_budget / per_share_risk)
        if quantity <= 0:
            return _rejected(RiskRejection.ZERO_QUANTITY)
        return Stage7RiskDecision(
            approved=True,
            reason="APPROVED",
            risk_budget=risk_budget,
            per_share_risk=per_share_risk,
            quantity=quantity,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
        )


class Stage7ExecutionService:
    """Turn selected Stage 6 intents into PAPER orders after reconciliation."""

    def __init__(
        self,
        *,
        run: RunConfig,
        expected_account: str,
        broker: ExecutionBroker,
        ledger: ExecutionLedger,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._run = run
        self._expected_account = expected_account
        self._broker = broker
        self._ledger = ledger
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self._reconciled_epoch: int | None = None

    async def reconcile(self) -> ReconciliationResult:
        """Compare broker orders/positions/fills with local execution state."""

        if self._run.environment is Environment.LIVE:
            return self._reconciliation_failure("LIVE execution is disabled in Stage 7")
        if not self._broker.is_connected:
            return self._reconciliation_failure("broker is disconnected")
        account_state = await self._broker.account_state()
        mismatch = self._account_mismatch(account_state)
        if mismatch is not None:
            return self._reconciliation_failure(mismatch)

        open_orders = await self._broker.read_open_orders()
        broker_statuses = await self._broker.read_order_statuses()
        broker_fills = await self._broker.read_fills()
        broker_positions = await self._broker.read_positions()
        known_ids = self._ledger.known_order_ids(self._run.environment, self._expected_account)
        problems: list[str] = []
        for status in broker_statuses:
            if not self._ledger.record_order_status(status):
                problems.append(f"unexpected broker order status {status.order_id}")
        for fill in broker_fills:
            if self._ledger.order_role(fill.environment, fill.account, fill.order_id) is None:
                problems.append(f"unexpected broker fill {fill.execution_id}")
                continue
            self._ledger.record_fill(fill)
        for order in open_orders:
            if (
                order.environment is not self._run.environment
                or order.account != self._expected_account
                or order.order_id not in known_ids
            ):
                problems.append(f"unexpected broker open order {order.order_id}")

        local_positions = self._ledger.positions(self._run.environment, self._expected_account)
        local_by_con_id = {position.con_id: position.quantity for position in local_positions}
        broker_by_con_id = {
            position.con_id: position.quantity
            for position in broker_positions
            if position.account == self._expected_account and position.quantity != 0.0
        }
        for con_id, quantity in broker_by_con_id.items():
            if con_id not in local_by_con_id:
                problems.append(f"unexpected broker position {con_id} quantity={quantity}")
            elif abs(local_by_con_id[con_id] - quantity) > 1e-9:
                problems.append(f"broker/local position mismatch for {con_id}")
        for con_id in local_by_con_id.keys() - broker_by_con_id.keys():
            problems.append(f"local position missing at broker for {con_id}")

        open_ids = {order.order_id for order in open_orders}
        for record in self._ledger.active_records(self._run.environment, self._expected_account):
            ids = {
                value
                for value in (
                    record.parent_order_id,
                    record.stop_order_id,
                    record.target_order_id,
                )
                if value is not None
            }
            local_quantity = local_by_con_id.get(record.con_id, 0.0)
            if not ids:
                problems.append(f"local plan {record.order_plan_id} has no broker identity")
            elif not ids.intersection(open_ids) and local_quantity == 0.0:
                problems.append(f"local order state is unresolved for {record.order_plan_id}")

        if problems:
            return self._reconciliation_failure("; ".join(sorted(set(problems))))
        self._reconciled_epoch = self._broker.connection_epoch
        return ReconciliationResult(
            ok=True,
            code=ExecutionResultCode.SUBMITTED,
            detail="broker and local execution state reconciled",
        )

    async def execute(
        self,
        order_intent: StrategySignal,
        instrument: QualifiedInstrument,
        *,
        diagnostic: bool = False,
    ) -> ExecutionAttempt:
        """Submit one selected intent, returning a candidate-local outcome."""

        if self._run.environment is Environment.LIVE:
            return _attempt(ExecutionResultCode.LIVE_EXECUTION_DISABLED)
        if not self._broker.is_connected:
            self._reconciled_epoch = None
            return _attempt(ExecutionResultCode.BROKER_DISCONNECTED)
        account_state = await self._broker.account_state()
        mismatch = self._account_mismatch(account_state)
        if mismatch is not None:
            return _attempt(ExecutionResultCode.ACCOUNT_OR_ENVIRONMENT_MISMATCH, mismatch)
        if self._ledger.has_signal(order_intent.signal_id):
            return _attempt(ExecutionResultCode.DUPLICATE_ORDER_BLOCKED)
        if self._reconciled_epoch != self._broker.connection_epoch:
            return _attempt(ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED)
        if (
            order_intent.run_id != self._run.run_id
            or instrument.con_id != order_intent.underlying_con_id
        ):
            return _attempt(
                ExecutionResultCode.ACCOUNT_OR_ENVIRONMENT_MISMATCH,
                "run or qualified instrument does not match the Stage 6 intent",
            )

        risk = Stage7RiskEngine().evaluate(
            order_intent=order_intent,
            account_state=account_state,
            risk_config=self._run.risk,
        )
        if not risk.approved:
            return _attempt(ExecutionResultCode(risk.reason), risk.reason)
        minimum_tick = await self._broker.minimum_tick(instrument)
        try:
            plan = build_order_plan(
                order_intent=order_intent,
                risk_decision=risk,
                environment=self._run.environment,
                minimum_tick=minimum_tick,
                created_at=self._clock(),
                diagnostic=diagnostic,
            )
        except ValueError as exc:
            return _attempt(ExecutionResultCode.INVALID_STOP_DISTANCE, str(exc))
        if not self._ledger.reserve(plan, expected_account=self._expected_account):
            return _attempt(ExecutionResultCode.DUPLICATE_ORDER_BLOCKED)
        self._ledger.mark_submitting(plan.order_plan_id)
        try:
            order_ids = await self._broker.submit_protected_order(plan, instrument)
            self._ledger.record_submission(
                plan.order_plan_id, order_ids, actual_account=account_state.account
            )
        except Exception as exc:
            reason = str(exc).strip() or "broker rejected protected order"
            self._ledger.record_rejection(plan.order_plan_id, reason)
            return ExecutionAttempt(
                code=ExecutionResultCode.BROKER_REJECTED,
                detail=reason,
                order_plan=plan,
            )
        return ExecutionAttempt(
            code=ExecutionResultCode.SUBMITTED,
            detail="protected PAPER order submitted",
            order_plan=plan,
            order_ids=order_ids,
        )

    async def refresh_broker_state(self) -> ReconciliationResult:
        """Apply normalized broker status/fill updates after submission."""

        if not self._broker.is_connected:
            return self._reconciliation_failure("broker is disconnected")
        unknown: list[str] = []
        for status in await self._broker.read_order_statuses():
            if not self._ledger.record_order_status(status):
                unknown.append(f"unknown broker order status {status.order_id}")
        for fill in await self._broker.read_fills():
            role = self._ledger.order_role(fill.environment, fill.account, fill.order_id)
            if role is None:
                unknown.append(f"unknown broker fill {fill.execution_id}")
            else:
                self._ledger.record_fill(fill)
        if unknown:
            return self._reconciliation_failure("; ".join(unknown))
        return ReconciliationResult(
            ok=True,
            code=ExecutionResultCode.SUBMITTED,
            detail="broker status and fills applied",
        )

    def record_fill(self, fill: BrokerFill) -> bool:
        """Apply a normalized callback once; unknown fills force reconciliation."""

        accepted = self._ledger.record_fill(fill)
        if (
            not accepted
            and self._ledger.order_role(fill.environment, fill.account, fill.order_id) is None
        ):
            self._reconciled_epoch = None
        return accepted

    def _account_mismatch(self, state: BrokerAccountState) -> str | None:
        if (
            not state.connected
            or state.environment is not Environment.PAPER
            or self._broker.environment is not Environment.PAPER
            or self._run.environment is not Environment.PAPER
            or state.account != self._expected_account
            or self._broker.account != self._expected_account
        ):
            return (
                f"run={self._run.environment.value} broker={self._broker.environment.value} "
                f"expected_account={self._expected_account} actual_account={state.account}"
            )
        return None

    def _reconciliation_failure(self, detail: str) -> ReconciliationResult:
        self._reconciled_epoch = None
        return ReconciliationResult(
            ok=False,
            code=ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED,
            detail=detail,
        )


def build_order_plan(
    *,
    order_intent: StrategySignal,
    risk_decision: Stage7RiskDecision,
    environment: Environment,
    minimum_tick: float,
    created_at: datetime,
    diagnostic: bool = False,
) -> OrderPlan:
    """Map one approved first-strategy intent into a protected SHORT plan."""

    if not isfinite(minimum_tick) or minimum_tick <= 0.0:
        raise ValueError("instrument minimum tick must be finite and positive")
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("order plan creation timestamp must be timezone-aware")
    if not risk_decision.approved or risk_decision.quantity <= 0:
        raise ValueError("order plan requires an approved positive risk decision")
    if order_intent.side != "SHORT" or risk_decision.stop_price <= risk_decision.entry_price:
        raise ValueError("SHORT protection requires a stop above entry")
    if risk_decision.target_price <= 0.0 or risk_decision.target_price >= risk_decision.entry_price:
        raise ValueError("SHORT protection requires a positive target below entry")

    stop = _to_tick(risk_decision.stop_price, minimum_tick, ROUND_FLOOR)
    target = _to_tick(risk_decision.target_price, minimum_tick, ROUND_CEILING)
    if stop <= risk_decision.entry_price or target >= risk_decision.entry_price:
        raise ValueError("SHORT protection became inverted at the instrument minimum tick")
    identity = "|".join(
        (
            "STAGE7_ORDER_PLAN_V1",
            order_intent.signal_id,
            environment.value,
            str(order_intent.underlying_con_id),
        )
    )
    return OrderPlan(
        order_plan_id=hashlib.sha256(identity.encode()).hexdigest()[:32],
        run_id=order_intent.run_id,
        signal_id=order_intent.signal_id,
        strategy_id=order_intent.strategy_id,
        strategy_version=order_intent.strategy_version,
        con_id=int(order_intent.underlying_con_id or 0),
        symbol=order_intent.symbol,
        side=OrderAction.SELL,
        quantity=risk_decision.quantity,
        entry_order_type=EntryOrderType.MARKET,
        entry_reference=risk_decision.entry_price,
        stop_price=stop,
        target_price=target,
        environment=environment,
        created_at=created_at,
        diagnostic=diagnostic,
    )


def _to_tick(value: float, minimum_tick: float, rounding: str) -> float:
    price = Decimal(str(value))
    tick = Decimal(str(minimum_tick))
    return float((price / tick).to_integral_value(rounding=rounding) * tick)


def _rejected(reason: RiskRejection) -> Stage7RiskDecision:
    return Stage7RiskDecision(approved=False, reason=reason.value)


def _attempt(code: ExecutionResultCode, detail: str | None = None) -> ExecutionAttempt:
    return ExecutionAttempt(code=code, detail=detail or code.value)
