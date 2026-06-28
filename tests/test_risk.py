from stocker_execution.orders import OrderSide, ProposedOrder
from stocker_execution.risk import RiskLimits, RiskState, check_order_allowed


def test_trading_disabled_blocks_all_orders() -> None:
    order = ProposedOrder(symbol="AAPL", side=OrderSide.BUY, quantity=10, estimated_price=100)
    limits = RiskLimits(trading_enabled=False)

    decision = check_order_allowed(order, limits, RiskState())

    assert decision.allowed is False
    assert "disabled" in decision.reason.lower()


def test_order_larger_than_max_order_size_is_rejected() -> None:
    order = ProposedOrder(symbol="AAPL", side=OrderSide.BUY, quantity=11, estimated_price=100)
    limits = RiskLimits(trading_enabled=True, max_order_size=1000)

    decision = check_order_allowed(order, limits, RiskState())

    assert decision.allowed is False
    assert "order size" in decision.reason.lower()


def test_order_that_exceeds_daily_order_count_is_rejected() -> None:
    order = ProposedOrder(symbol="AAPL", side=OrderSide.BUY, quantity=1, estimated_price=100)
    limits = RiskLimits(
        trading_enabled=True,
        max_position_size=1000,
        max_order_size=1000,
        max_daily_loss=500,
        max_orders_per_day=2,
    )
    state = RiskState(orders_placed_today=2)

    decision = check_order_allowed(order, limits, state)

    assert decision.allowed is False
    assert "orders per day" in decision.reason.lower()


def test_valid_order_is_allowed() -> None:
    order = ProposedOrder(symbol="AAPL", side=OrderSide.BUY, quantity=5, estimated_price=100)
    limits = RiskLimits(
        trading_enabled=True,
        max_position_size=1000,
        max_order_size=1000,
        max_daily_loss=500,
        max_orders_per_day=10,
    )
    state = RiskState(
        current_positions={"AAPL": 100}, realized_pnl_today=-100, orders_placed_today=1
    )

    decision = check_order_allowed(order, limits, state)

    assert decision.allowed is True
    assert decision.reason == "allowed"
