"""Hard pre-trade risk checks."""

from pydantic import BaseModel, Field

from stocker_execution.orders import OrderSide, ProposedOrder


class RiskLimits(BaseModel):
    """Risk limits that must pass before an order can be submitted."""

    max_position_size: float = Field(default=0.0, ge=0.0)
    max_order_size: float = Field(default=0.0, ge=0.0)
    max_daily_loss: float = Field(default=0.0, ge=0.0)
    max_orders_per_day: int = Field(default=0, ge=0)
    trading_enabled: bool = False


class RiskState(BaseModel):
    """State needed to evaluate a proposed order."""

    current_positions: dict[str, float] = Field(default_factory=dict)
    realized_pnl_today: float = 0.0
    orders_placed_today: int = 0


class RiskDecision(BaseModel):
    """Risk check result."""

    allowed: bool
    reason: str


def _projected_position_notional(order: ProposedOrder, state: RiskState) -> float:
    current = state.current_positions.get(order.symbol, 0.0)
    if order.side is OrderSide.BUY:
        return current + order.notional
    return current - order.notional


def check_order_allowed(order: ProposedOrder, limits: RiskLimits, state: RiskState) -> RiskDecision:
    """Return whether a proposed order passes all hard risk limits."""

    if not limits.trading_enabled:
        return RiskDecision(allowed=False, reason="trading disabled")
    if limits.max_order_size <= 0:
        return RiskDecision(allowed=False, reason="max order size is not configured")
    if order.notional > limits.max_order_size:
        return RiskDecision(allowed=False, reason="order size exceeds max order size")
    if limits.max_daily_loss <= 0:
        return RiskDecision(allowed=False, reason="max daily loss is not configured")
    if state.realized_pnl_today <= -limits.max_daily_loss:
        return RiskDecision(allowed=False, reason="max daily loss reached")
    if limits.max_orders_per_day <= 0:
        return RiskDecision(allowed=False, reason="max orders per day is not configured")
    if state.orders_placed_today >= limits.max_orders_per_day:
        return RiskDecision(allowed=False, reason="max orders per day reached")
    if limits.max_position_size <= 0:
        return RiskDecision(allowed=False, reason="max position size is not configured")
    if abs(_projected_position_notional(order, state)) > limits.max_position_size:
        return RiskDecision(allowed=False, reason="max position size exceeded")
    return RiskDecision(allowed=True, reason="allowed")
