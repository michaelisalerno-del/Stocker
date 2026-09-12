"""Stage 7 risk, planning, and environment-scoped execution orchestration."""

import asyncio
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import StrEnum
from math import floor, isfinite
from typing import Protocol

from stocker_core.runs import Environment, RunConfig, RunRiskConfig
from stocker_execution.discovery import DiscoveryFx
from stocker_execution.execution_currency import execution_valuation
from stocker_execution.execution_ledger import AdmissionRejected, ExecutionLedger
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
    StockExecutionRules,
)
from stocker_execution.ibkr import BrokerSession, CurrentQuote, QualifiedInstrument
from stocker_execution.session_hard_structure_d import (
    EntryBar,
    SessionHardStructureDStrategy,
    SignalStatus,
    StrategySignal,
    nominal_exit_prices,
)

MAX_ENTRY_SIGNAL_AGE = timedelta(seconds=60)
MAX_ENTRY_QUOTE_AGE = timedelta(seconds=5)
ENTRY_ORDER_LIFETIME = timedelta(seconds=5)


class RiskRejection(StrEnum):
    INVALID_RISK_CONFIG = "INVALID_RISK_CONFIG"
    INVALID_STOP_DISTANCE = "INVALID_STOP_DISTANCE"
    ZERO_QUANTITY = "ZERO_QUANTITY"
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    CAPACITY_REACHED = "CAPACITY_REACHED"
    ACCOUNT_STATE_UNAVAILABLE = "ACCOUNT_STATE_UNAVAILABLE"


class ExecutionResultCode(StrEnum):
    PENDING_ENTRY_CAPACITY_UNVERIFIED = "PENDING_ENTRY_CAPACITY_UNVERIFIED"
    EXPOSURE_POLICY_REQUIRED = "EXPOSURE_POLICY_REQUIRED"
    SUBMITTED = "SUBMITTED"
    STALE_SIGNAL = "STALE_SIGNAL"
    ENTRY_QUOTE_UNAVAILABLE = "ENTRY_QUOTE_UNAVAILABLE"
    ENTRY_PRICE_MOVED = "ENTRY_PRICE_MOVED"
    EXECUTION_ENVIRONMENT_UNAVAILABLE = "EXECUTION_ENVIRONMENT_UNAVAILABLE"
    ACCOUNT_OR_ENVIRONMENT_MISMATCH = "ACCOUNT_OR_ENVIRONMENT_MISMATCH"
    BROKER_DISCONNECTED = "BROKER_DISCONNECTED"
    EXECUTION_RECONCILIATION_REQUIRED = "EXECUTION_RECONCILIATION_REQUIRED"
    DUPLICATE_ORDER_BLOCKED = "DUPLICATE_ORDER_BLOCKED"
    BROKER_REJECTED = "BROKER_REJECTED"
    ORDER_PLAN_UNAVAILABLE = "ORDER_PLAN_UNAVAILABLE"
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
    run_id: str
    signal_id: str
    environment: Environment
    expected_account: str
    actual_account: str | None
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

    async def connect(self) -> BrokerSession: ...

    def disconnect(self) -> None: ...

    async def account_state(self, *, fresh: bool = False) -> BrokerAccountState: ...

    async def minimum_tick(self, instrument: QualifiedInstrument) -> float: ...

    async def entry_quote(self, instrument: QualifiedInstrument) -> CurrentQuote: ...

    async def discovery_fx(self, currency: str) -> DiscoveryFx: ...

    async def stock_execution_rules(
        self, instrument: QualifiedInstrument
    ) -> StockExecutionRules: ...

    async def shortable_quantity(self, instrument: QualifiedInstrument) -> float: ...

    async def check_order_capacity(
        self, plan: OrderPlan, instrument: QualifiedInstrument
    ) -> None: ...

    async def submit_protected_order(
        self, plan: OrderPlan, instrument: QualifiedInstrument
    ) -> BrokerOrderIds: ...

    async def read_open_orders(self) -> tuple[BrokerOpenOrder, ...]: ...

    async def read_fills(self) -> tuple[BrokerFill, ...]: ...

    async def read_positions(self) -> tuple[BrokerPosition, ...]: ...

    async def read_order_statuses(
        self, *, include_completed: bool = True
    ) -> tuple[BrokerOrderStatus, ...]: ...


@dataclass(frozen=True, slots=True)
class ExecutionDestination:
    """One explicit broker environment/account destination."""

    environment: Environment
    expected_account: str
    broker: ExecutionBroker


class ExecutionEnvironmentUnavailableError(RuntimeError):
    """Raised when a run requests an unconfigured execution environment."""


class ExecutionRouter:
    """Resolve a run environment without cross-environment fallback."""

    def __init__(self, destinations: Sequence[ExecutionDestination]) -> None:
        self._destinations: dict[Environment, ExecutionDestination] = {}
        broker_ids: set[int] = set()
        expected_accounts: set[str] = set()
        for destination in destinations:
            if not destination.expected_account.strip():
                raise ValueError(
                    f"{destination.environment.value} execution requires expected_account"
                )
            if destination.environment is not destination.broker.environment:
                raise ValueError("ACCOUNT_OR_ENVIRONMENT_MISMATCH")
            if destination.environment in self._destinations:
                raise ValueError(
                    f"Duplicate execution environment: {destination.environment.value}"
                )
            if id(destination.broker) in broker_ids:
                raise ValueError("PAPER and LIVE require distinct broker session objects")
            if destination.expected_account in expected_accounts:
                raise ValueError("PAPER and LIVE require distinct expected accounts")
            self._destinations[destination.environment] = destination
            broker_ids.add(id(destination.broker))
            expected_accounts.add(destination.expected_account)

    @property
    def environments(self) -> tuple[Environment, ...]:
        return tuple(self._destinations)

    def for_environment(self, environment: Environment) -> ExecutionDestination:
        try:
            return self._destinations[environment]
        except KeyError as exc:
            raise ExecutionEnvironmentUnavailableError(
                f"EXECUTION_ENVIRONMENT_UNAVAILABLE: {environment.value}"
            ) from exc


class Stage7RiskEngine:
    """Size one Stage 6 intent without making broker calls."""

    def evaluate(
        self,
        *,
        order_intent: StrategySignal,
        account_state: BrokerAccountState,
        risk_config: RunRiskConfig | None,
        account_per_price_unit: float = 1.0,
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
        explicit_exits = (
            order_intent.stop_price is not None and order_intent.target_price is not None
        )
        if (
            order_intent.status is not SignalStatus.ENTRY_TRIGGERED
            or not order_intent.selected
            or order_intent.side not in {"SHORT", "LONG"}
            or entry is None
            or not isfinite(entry)
            or entry <= 0.0
            or (
                not explicit_exits
                and (
                    m_price is None
                    or not isfinite(m_price)
                    or m_price <= 0.0
                    or not isfinite(order_intent.stop_distance_m)
                    or order_intent.stop_distance_m <= 0.0
                    or not isfinite(order_intent.target_distance_m)
                    or order_intent.target_distance_m <= 0.0
                )
            )
        ):
            return _rejected(RiskRejection.INVALID_STOP_DISTANCE)

        stop, target = nominal_exit_prices(order_intent)
        per_share_risk = abs(entry - stop)
        if (
            not isfinite(stop)
            or not isfinite(target)
            or (stop <= entry if order_intent.side == "SHORT" else stop >= entry)
            or target <= 0.0
            or (target >= entry if order_intent.side == "SHORT" else target <= entry)
            or per_share_risk <= 0.0
        ):
            return _rejected(RiskRejection.INVALID_STOP_DISTANCE)

        risk_budget = equity * risk_fraction
        if not isfinite(account_per_price_unit) or account_per_price_unit <= 0:
            return _rejected(RiskRejection.ACCOUNT_STATE_UNAVAILABLE)
        quantity = floor(risk_budget / (per_share_risk * account_per_price_unit))
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
    """Turn selected Stage 6 intents into orders for one explicit destination."""

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
        self._last_account_state: BrokerAccountState | None = None

    @property
    def run_environment(self) -> Environment:
        return self._run.environment

    @property
    def run_config(self) -> RunConfig:
        """Return the immutable config that owns this execution lineage."""

        return self._run

    @property
    def is_reconciled(self) -> bool:
        return (
            self._broker.is_connected
            and self._broker.account == self._expected_account
            and self._broker.environment is self._run.environment
            and self._reconciled_epoch == self._broker.connection_epoch
        )

    @property
    def last_account_state(self) -> BrokerAccountState | None:
        """Return the authoritative account state read during reconciliation."""

        return self._last_account_state

    def update_run_config(self, run: RunConfig) -> None:
        """Apply future-only risk changes without resetting reconciliation state."""

        if (
            run.run_id != self._run.run_id
            or run.universe != self._run.universe
            or run.strategy != self._run.strategy
            or run.environment is not self._run.environment
            or run.session != self._run.session
        ):
            raise ValueError("Stage 7 hot apply accepts risk/enabled changes only")
        self._run = run

    def adopt_reconciliation(self, source: "Stage7ExecutionService") -> None:
        """Reuse one account-wide reconciliation across runs on the same broker session."""

        if (
            source._broker is not self._broker
            or source._expected_account != self._expected_account
            or source.run_environment is not self.run_environment
            or source._reconciled_epoch != self._broker.connection_epoch
            or source._last_account_state is None
        ):
            raise ValueError("shared reconciliation source does not match this execution scope")
        self._last_account_state = source._last_account_state
        self._reconciled_epoch = source._reconciled_epoch

    def invalidate_reconciliation(self) -> None:
        """Require a fresh broker snapshot before this run may submit another order."""

        self._last_account_state = None
        self._reconciled_epoch = None

    async def reconcile(self) -> ReconciliationResult:
        """Compare broker orders/positions/fills with local execution state."""

        self._last_account_state = None
        if not self._broker.is_connected:
            return self._reconciliation_failure("broker is disconnected")
        try:
            account_state = await self._broker.account_state()
        except Exception as exc:
            return self._reconciliation_failure(f"account state unavailable: {exc}")
        self._last_account_state = account_state
        mismatch = self._account_mismatch(account_state)
        if mismatch is not None:
            return self._reconciliation_failure(mismatch)

        try:
            open_orders = await self._broker.read_open_orders()
            # Completed history is needed to recover unfinished plans across a
            # connection gap. A flat, settled ledger needs only current broker
            # evidence; orders placed on a reconciled connection have live callbacks.
            # Scope this to the whole account, including sibling runs.
            include_completed = self._reconciled_epoch != self._broker.connection_epoch and bool(
                self._ledger.active_records(self._run.environment, self._expected_account)
            )
            broker_statuses = await self._broker.read_order_statuses(
                include_completed=include_completed
            )
            broker_fills = await self._broker.read_fills()
            broker_positions = await self._broker.read_positions()
        except Exception as exc:
            return self._reconciliation_failure(f"broker state unavailable: {exc}")

        self._ledger.replace_broker_snapshot(
            environment=self._run.environment,
            account=self._expected_account,
            positions=broker_positions,
            open_orders=open_orders,
            observed_at=self._clock(),
        )

        known_ids = set(self._ledger.known_order_ids(self._run.environment, self._expected_account))
        problems: list[str] = []
        for order in open_orders:
            if (
                order.environment is not self._run.environment
                or order.account != self._expected_account
            ):
                problems.append(f"unexpected broker open order {order.order_id}")
            elif order.order_id not in known_ids:
                if self._ledger.recover_open_order(order):
                    known_ids.add(order.order_id)
                else:
                    problems.append(f"unexpected broker open order {order.order_id}")
        for status in broker_statuses:
            recorded = self._ledger.record_order_status(status)
            if not recorded and status.status not in {
                OrderLifecycle.FILLED,
                OrderLifecycle.CANCELLED,
                OrderLifecycle.REJECTED,
                OrderLifecycle.CLOSED,
            }:
                problems.append(f"unexpected broker order status {status.order_id}")
        for fill in broker_fills:
            role = self._ledger.order_role(fill.environment, fill.account, fill.order_id)
            if role is None and not self._ledger.recover_entry_fill_order(fill):
                problems.append(f"unexpected broker fill {fill.execution_id}")
                continue
            self._ledger.record_fill(fill)
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
            reported = record.broker_reported_entry_filled
            if reported is None or reported > record.filled_quantity:
                problems.append(f"entry execution details unresolved for {record.order_plan_id}")
            ids = {
                value
                for value in (
                    record.parent_order_id,
                    record.stop_order_id,
                    record.target_order_id,
                    record.timeout_order_id,
                )
                if value is not None
            }
            local_quantity = local_by_con_id.get(record.con_id, 0.0)
            if not ids:
                problems.append(f"local plan {record.order_plan_id} has no broker identity")
            elif record.deadline is not None and record.timeout_order_id is None:
                problems.append(f"missing method deadline order for {record.order_plan_id}")
            elif local_quantity != 0.0 and not (
                {record.stop_order_id, record.target_order_id}
                | ({record.timeout_order_id} if record.timeout_order_id is not None else set())
            ).issubset(open_ids):
                problems.append(
                    f"position {record.con_id} missing protective stop/target "
                    f"for {record.order_plan_id}"
                )
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

        admission_run = self._run
        if not self._broker.is_connected:
            self._reconciled_epoch = None
            return self._outcome(order_intent.signal_id, ExecutionResultCode.BROKER_DISCONNECTED)
        account_revision = self._ledger.exposure_revision(
            self._run.environment, self._expected_account
        )
        try:
            account_state = await self._broker.account_state(fresh=True)
        except Exception as exc:
            self._reconciled_epoch = None
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ACCOUNT_STATE_UNAVAILABLE,
                f"account state unavailable: {exc}",
            )
        mismatch = self._account_mismatch(account_state)
        if mismatch is not None:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ACCOUNT_OR_ENVIRONMENT_MISMATCH,
                mismatch,
                actual_account=account_state.account,
            )
        if self._ledger.has_signal(order_intent.signal_id):
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.DUPLICATE_ORDER_BLOCKED,
                actual_account=account_state.account,
            )
        if self._reconciled_epoch != self._broker.connection_epoch:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED,
                actual_account=account_state.account,
            )
        if (
            order_intent.run_id != self._run.run_id
            or instrument.con_id != order_intent.underlying_con_id
        ):
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ACCOUNT_OR_ENVIRONMENT_MISMATCH,
                "run or qualified instrument does not match the Stage 6 intent",
                actual_account=account_state.account,
            )
        try:
            valuation = await asyncio.wait_for(
                execution_valuation(self._broker, instrument, account_state.currency, self._clock),
                timeout=4.0,
            )
        except Exception as exc:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ACCOUNT_STATE_UNAVAILABLE,
                f"Execution currency valuation unavailable: {exc}",
                actual_account=account_state.account,
            )
        risk = Stage7RiskEngine().evaluate(
            order_intent=order_intent,
            account_state=account_state,
            risk_config=self._run.risk,
            account_per_price_unit=valuation.account_per_price_unit,
        )
        if self._run.method_spec_hash is not None and (
            order_intent.method_spec_hash != self._run.method_spec_hash
            or order_intent.strategy_id != self._run.strategy_id
            or order_intent.strategy_version != self._run.strategy_version
            or order_intent.market_id != self._run.market_id
        ):
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ORDER_PLAN_UNAVAILABLE,
                "Signal provenance does not match the run's frozen method",
                actual_account=account_state.account,
            )
        if not risk.approved:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode(risk.reason),
                risk.reason,
                actual_account=account_state.account,
            )
        try:
            minimum_tick = await self._broker.minimum_tick(instrument)
        except Exception as exc:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ORDER_PLAN_UNAVAILABLE,
                str(exc).strip() or "instrument minimum tick is unavailable",
                actual_account=account_state.account,
            )
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
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.INVALID_STOP_DISTANCE,
                str(exc),
                actual_account=account_state.account,
            )
        # A historical intrabar touch does not establish a currently executable price.
        if not _fresh_entry_signal(order_intent, self._clock()):
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.STALE_SIGNAL,
                "entry signal is missing, future-dated, or more than 60 seconds old",
                actual_account=account_state.account,
            )
        try:
            quote = await asyncio.wait_for(self._broker.entry_quote(instrument), timeout=4.0)
        except Exception as exc:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ENTRY_QUOTE_UNAVAILABLE,
                f"fresh live bid/ask unavailable: {exc}",
                actual_account=account_state.account,
            )
        checked_at = self._clock()
        if self._run is not admission_run:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED,
                "Run configuration changed during entry preparation",
                actual_account=account_state.account,
            )
        if not self._run.enabled:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED,
                "new entries paused",
                actual_account=account_state.account,
            )
        if not self._broker.is_connected:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.BROKER_DISCONNECTED,
                actual_account=account_state.account,
            )
        if (
            self._broker.account != self._expected_account
            or self._broker.environment is not self._run.environment
        ):
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ACCOUNT_OR_ENVIRONMENT_MISMATCH,
                "execution destination changed while awaiting the quote",
                actual_account=self._broker.account,
            )
        if self._reconciled_epoch != self._broker.connection_epoch:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED,
                "broker connection changed while awaiting the quote",
                actual_account=account_state.account,
            )
        if not _fresh_entry_signal(order_intent, checked_at):
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.STALE_SIGNAL,
                "entry signal expired while awaiting a live quote",
                actual_account=account_state.account,
            )
        if not _valid_entry_quote(quote, instrument, checked_at):
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ENTRY_QUOTE_UNAVAILABLE,
                "live, correctly identified, uncrossed bid/ask no older than 5 seconds required",
                actual_account=account_state.account,
            )
        long = order_intent.side == "LONG"
        limit = _to_tick(plan.entry_reference, minimum_tick, ROUND_FLOOR if long else ROUND_CEILING)
        assert quote.bid is not None and quote.ask is not None
        if (
            (quote.ask > limit or quote.bid <= plan.stop_price)
            if long
            else (quote.bid < limit or quote.ask >= plan.stop_price)
        ):
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ENTRY_PRICE_MOVED,
                f"{order_intent.side}: bid={quote.bid:g} ask={quote.ask:g}; "
                f"limit={limit:g} stop={plan.stop_price:g}; entry is no longer executable",
                actual_account=account_state.account,
            )
        assert order_intent.signal_timestamp is not None
        plan = replace(
            plan,
            created_at=checked_at,
            account_currency=valuation.account_currency,
            price_currency=valuation.price_currency,
            price_unit=valuation.price_unit,
            account_per_price_unit=valuation.account_per_price_unit,
            fx_observed_at=valuation.fx_observed_at,
            fx_evidence=valuation.fx_evidence,
            minimum_quantity=valuation.minimum_quantity,
            quantity_increment=valuation.quantity_increment,
            entry_order_type=EntryOrderType.LIMIT,
            entry_limit_price=limit,
            entry_expires_at=min(
                checked_at + ENTRY_ORDER_LIFETIME,
                _entry_deadline(order_intent),
                order_intent.t0 + timedelta(minutes=5),
            ),
        )
        exposure_limit = self._run.risk.max_gross_notional if self._run.risk else None
        if exposure_limit is None:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.EXPOSURE_POLICY_REQUIRED,
                "Set risk.max_gross_notional in the verified account currency before new entries",
                actual_account=account_state.account,
            )
        if (
            account_state.gross_position_value is None
            or not isfinite(account_state.gross_position_value)
            or account_state.gross_position_value < 0
        ):
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ACCOUNT_STATE_UNAVAILABLE,
                "Verified account-currency gross position value required",
                actual_account=account_state.account,
            )
        try:
            valuation.validate(self._clock())
            reserved = self._ledger.reserve(
                plan,
                expected_account=self._expected_account,
                positions=account_state.positions,
                max_positions=self._run.risk.max_concurrent_positions if self._run.risk else None,
                max_gross_notional=exposure_limit,
                broker_gross_notional=account_state.gross_position_value,
                account_revision=account_revision,
                require_settled_entries=True,
            )
        except (AdmissionRejected, ValueError) as exc:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode(str(exc))
                if isinstance(exc, AdmissionRejected)
                else ExecutionResultCode.ACCOUNT_STATE_UNAVAILABLE,
                str(exc),
                actual_account=account_state.account,
            )
        if not reserved:
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.DUPLICATE_ORDER_BLOCKED,
                actual_account=account_state.account,
            )
        admitted = self._ledger.get(plan.order_plan_id)
        assert admitted is not None
        plan = replace(
            plan,
            quantity=admitted.intended_quantity,
            risk_derived_quantity=admitted.risk_derived_quantity,
            sizing_reason=admitted.sizing_reason,
        )
        try:
            if (
                self._run.method_spec is not None
                and plan.side is OrderAction.SELL
                and self._run.method_spec["execution"].get("shortability") == "Required for SHORT"
            ):
                available = await self._broker.shortable_quantity(instrument)
                if not isfinite(available) or available < plan.quantity:
                    raise ValueError("METHOD_SHORTABILITY_UNAVAILABLE")
            await self._broker.check_order_capacity(plan, instrument)
            now = self._clock()
            valuation.validate(now)
            if self._run is not admission_run:
                raise ValueError("Run configuration changed during credit preview")
            if (
                not self._broker.is_connected
                or self._broker.account != self._expected_account
                or self._broker.environment is not self._run.environment
                or self._reconciled_epoch != self._broker.connection_epoch
            ):
                raise ValueError("Execution destination changed during credit preview")
            if not self._run.enabled or not _fresh_entry_signal(order_intent, now):
                raise ValueError("Entry paused or signal expired during credit preview")
            if plan.entry_expires_at is None or now >= plan.entry_expires_at:
                raise ValueError("Entry expired during credit preview")
            if not _valid_entry_quote(quote, instrument, now):
                raise ValueError("Quote expired during credit preview")
        except Exception as exc:
            # No submission has begun: releasing this commitment is supported.
            self._ledger.record_rejection(plan.order_plan_id, str(exc))
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.ACCOUNT_STATE_UNAVAILABLE,
                "Broker capacity preview unavailable or entry expired",
                actual_account=account_state.account,
                order_plan=plan,
            )
        self._ledger.mark_submitting(plan.order_plan_id)
        try:
            order_ids = await self._broker.submit_protected_order(plan, instrument)
        except Exception as exc:
            # An exception is not proof that nothing reached IBKR. Keep the
            # durable SUBMITTING reservation until broker evidence settles it.
            self._reconciled_epoch = None
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED,
                f"submission outcome unknown; reconciliation required: {exc}",
                actual_account=account_state.account,
                order_plan=plan,
            )
        try:
            self._ledger.record_submission(
                plan.order_plan_id, order_ids, actual_account=account_state.account
            )
        except Exception as exc:
            self._reconciled_epoch = None
            return self._outcome(
                order_intent.signal_id,
                ExecutionResultCode.EXECUTION_RECONCILIATION_REQUIRED,
                f"broker order submitted but ledger persistence failed: {exc}",
                actual_account=account_state.account,
                order_plan=plan,
                order_ids=order_ids,
            )
        return self._outcome(
            order_intent.signal_id,
            ExecutionResultCode.SUBMITTED,
            f"protected {self._run.environment.value} order submitted",
            actual_account=account_state.account,
            order_plan=plan,
            order_ids=order_ids,
        )

    async def execute_ready_intents(
        self,
        order_intents: Sequence[StrategySignal],
        instruments: Mapping[int, QualifiedInstrument],
    ) -> tuple[ExecutionAttempt, ...]:
        """Execute selected Stage 6 entry intents while isolating candidate failures."""

        results: list[ExecutionAttempt] = []
        for intent in order_intents:
            if intent.status is not SignalStatus.ENTRY_TRIGGERED or not intent.selected:
                continue
            instrument = (
                instruments.get(intent.underlying_con_id)
                if intent.underlying_con_id is not None
                else None
            )
            if instrument is None:
                results.append(
                    self._outcome(
                        intent.signal_id,
                        ExecutionResultCode.ORDER_PLAN_UNAVAILABLE,
                        "qualified instrument is unavailable",
                    )
                )
                continue
            results.append(await self.execute(intent, instrument))
        return tuple(results)

    async def refresh_broker_state(self) -> ReconciliationResult:
        """Re-read and reconcile all broker execution state."""

        return await self.reconcile()

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
            or state.environment is not self._run.environment
            or self._broker.environment is not self._run.environment
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

    def _outcome(
        self,
        signal_id: str,
        code: ExecutionResultCode,
        detail: str | None = None,
        *,
        actual_account: str | None = None,
        order_plan: OrderPlan | None = None,
        order_ids: BrokerOrderIds | None = None,
    ) -> ExecutionAttempt:
        resolved_detail = detail or code.value
        if actual_account is None:
            actual_account = self._known_broker_account()
        attempt = ExecutionAttempt(
            code=code,
            detail=resolved_detail,
            run_id=self._run.run_id,
            signal_id=signal_id,
            environment=self._run.environment,
            expected_account=self._expected_account,
            actual_account=actual_account,
            order_plan=order_plan,
            order_ids=order_ids,
        )
        self._ledger.record_attempt(
            run_id=attempt.run_id,
            signal_id=signal_id,
            environment=attempt.environment,
            expected_account=attempt.expected_account,
            actual_account=attempt.actual_account,
            result_code=attempt.code.value,
            detail=attempt.detail,
            attempted_at=self._clock(),
        )
        return attempt

    def _known_broker_account(self) -> str | None:
        try:
            account = self._broker.account
        except Exception:
            return None
        return account or None


class Stage7StrategyRuntime:
    """Direct first-strategy runtime seam from Stage 6 observations to execution."""

    def __init__(
        self,
        *,
        strategy: SessionHardStructureDStrategy,
        execution: Stage7ExecutionService,
    ) -> None:
        self._strategy = strategy
        self._execution = execution

    async def observe_and_execute(
        self,
        bars_by_con_id: Mapping[int, Sequence[EntryBar]],
        instruments: Mapping[int, QualifiedInstrument],
    ) -> tuple[ExecutionAttempt, ...]:
        """Advance Stage 6 first-touch state and execute only its selected outputs."""

        return await self.execute_observed(self.observe(bars_by_con_id), instruments)

    def observe(
        self,
        bars_by_con_id: Mapping[int, Sequence[EntryBar]],
    ) -> tuple[StrategySignal, ...]:
        return self._strategy.observe_entry_bars(bars_by_con_id)

    async def execute_observed(
        self,
        order_intents: Sequence[StrategySignal],
        instruments: Mapping[int, QualifiedInstrument],
    ) -> tuple[ExecutionAttempt, ...]:
        """Use the same Stage 7 path after runtime admission has completed."""
        return await self._execution.execute_ready_intents(order_intents, instruments)


# Backwards-compatible Stage 7 name; both environments use the same implementation.
Stage7PaperRuntime = Stage7StrategyRuntime


def build_order_plan(
    *,
    order_intent: StrategySignal,
    risk_decision: Stage7RiskDecision,
    environment: Environment,
    minimum_tick: float,
    created_at: datetime,
    diagnostic: bool = False,
) -> OrderPlan:
    """Map one approved method intent into its protected LONG or SHORT plan."""

    if not isfinite(minimum_tick) or minimum_tick <= 0.0:
        raise ValueError("instrument minimum tick must be finite and positive")
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("order plan creation timestamp must be timezone-aware")
    if not risk_decision.approved or risk_decision.quantity <= 0:
        raise ValueError("order plan requires an approved positive risk decision")
    long = order_intent.side == "LONG"
    direction = 1 if long else -1
    if (risk_decision.entry_price - risk_decision.stop_price) * direction <= 0:
        raise ValueError("Protective stop is on the wrong side of entry")
    if (
        risk_decision.target_price <= 0
        or (risk_decision.target_price - risk_decision.entry_price) * direction <= 0
    ):
        raise ValueError("Protective target is on the wrong side of entry")
    stop = _to_tick(risk_decision.stop_price, minimum_tick, ROUND_CEILING if long else ROUND_FLOOR)
    target = _to_tick(
        risk_decision.target_price, minimum_tick, ROUND_FLOOR if long else ROUND_CEILING
    )
    if (risk_decision.entry_price - stop) * direction <= 0 or (
        target - risk_decision.entry_price
    ) * direction <= 0:
        raise ValueError("Protection became inverted at the instrument minimum tick")
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
        side=OrderAction.BUY if long else OrderAction.SELL,
        quantity=risk_decision.quantity,
        entry_order_type=EntryOrderType.MARKET,
        entry_reference=risk_decision.entry_price,
        stop_price=stop,
        target_price=target,
        environment=environment,
        created_at=created_at,
        initial_risk_budget=risk_decision.risk_budget,
        per_share_initial_risk=risk_decision.per_share_risk,
        diagnostic=diagnostic,
        deadline=order_intent.deadline,
        market_id=order_intent.market_id,
        method_spec_hash=order_intent.method_spec_hash,
        method_stop_price=risk_decision.stop_price,
        method_target_price=risk_decision.target_price,
    )


def _to_tick(value: float, minimum_tick: float, rounding: str) -> float:
    price = Decimal(str(value))
    tick = Decimal(str(minimum_tick))
    return float((price / tick).to_integral_value(rounding=rounding) * tick)


def _fresh_entry_signal(signal: StrategySignal, now: datetime) -> bool:
    timestamp = signal.signal_timestamp
    return (
        timestamp is not None
        and timestamp.tzinfo is not None
        and timestamp.utcoffset() is not None
        and now.tzinfo is not None
        and now.utcoffset() is not None
        and signal.t0.tzinfo is not None
        and signal.t0.utcoffset() is not None
        and (
            signal.entry_timestamp is None
            or (
                signal.entry_timestamp.tzinfo is not None
                and signal.entry_timestamp.utcoffset() is not None
            )
        )
        and timestamp <= now < _entry_deadline(signal)
    )


def _entry_deadline(signal: StrategySignal) -> datetime:
    assert signal.signal_timestamp is not None
    # Gap-at-open timestamps name the bar start, but production observes complete
    # bars. Both gap and intrabar entries get at most one minute after observation.
    observed_at = signal.signal_timestamp
    if signal.method_spec_hash is None and signal.entry_timestamp is not None:
        observed_at = max(observed_at, signal.entry_timestamp + timedelta(minutes=1))
    return min(observed_at + MAX_ENTRY_SIGNAL_AGE, signal.t0 + timedelta(minutes=5))


def _valid_entry_quote(quote: CurrentQuote, instrument: QualifiedInstrument, now: datetime) -> bool:
    timestamp = quote.timestamp
    return (
        quote.con_id == instrument.con_id
        and quote.symbol == instrument.symbol
        and quote.market_data_type == 1
        and timestamp is not None
        and timestamp.tzinfo is not None
        and timestamp.utcoffset() is not None
        and timedelta(0) <= now - timestamp <= MAX_ENTRY_QUOTE_AGE
        and quote.bid is not None
        and quote.ask is not None
        and isfinite(quote.bid)
        and isfinite(quote.ask)
        and 0 < quote.bid <= quote.ask
    )


def _rejected(reason: RiskRejection) -> Stage7RiskDecision:
    return Stage7RiskDecision(approved=False, reason=reason.value)
