"""Small broker-normalized models used by Stage 7 execution."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from stocker_core.runs import Environment


class OrderAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class EntryOrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderLifecycle(StrEnum):
    PLANNED = "PLANNED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    CLOSED = "CLOSED"


class OrderRole(StrEnum):
    ENTRY = "ENTRY"
    STOP = "STOP"
    TARGET = "TARGET"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True, slots=True)
class BrokerOrderIds:
    parent: int
    stop: int
    target: int
    timeout: int | None = None


@dataclass(frozen=True, slots=True)
class OrderPlan:
    """One broker-independent protected execution plan."""

    order_plan_id: str
    run_id: str
    signal_id: str
    strategy_id: str
    strategy_version: str
    con_id: int
    symbol: str
    side: OrderAction
    quantity: int
    entry_order_type: EntryOrderType
    entry_reference: float
    stop_price: float
    target_price: float
    environment: Environment
    created_at: datetime
    initial_risk_budget: float | None = None
    per_share_initial_risk: float | None = None
    diagnostic: bool = False
    entry_limit_price: float | None = None
    entry_expires_at: datetime | None = None
    deadline: datetime | None = None
    market_id: str | None = None
    method_spec_hash: str | None = None


@dataclass(frozen=True, slots=True)
class BrokerFill:
    """One normalized IBKR execution, deduplicated within environment/account."""

    execution_id: str
    order_id: int
    account: str
    environment: Environment
    con_id: int
    symbol: str
    side: OrderAction
    quantity: float
    price: float
    executed_at: datetime
    commission: float | None = None
    order_plan_id: str = ""


@dataclass(frozen=True, slots=True)
class BrokerOpenOrder:
    """One normalized open broker order used during reconciliation."""

    order_id: int
    order_plan_id: str
    account: str
    environment: Environment
    con_id: int
    symbol: str
    role: OrderRole
    status: OrderLifecycle


@dataclass(frozen=True, slots=True)
class BrokerOrderStatus:
    """One normalized current or completed broker order status."""

    order_id: int
    order_plan_id: str
    account: str
    environment: Environment
    status: OrderLifecycle
    filled_quantity: float
    remaining_quantity: float
    reason: str = ""


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """One actual broker position, scoped to its account."""

    account: str
    con_id: int
    symbol: str
    quantity: float
    average_price: float


@dataclass(frozen=True, slots=True)
class BrokerAccountState:
    """Authoritative account inputs required by the pure risk layer."""

    environment: Environment
    account: str
    equity: float | None
    buying_power: float | None
    connected: bool
    positions: tuple[BrokerPosition, ...] = ()
