"""Minimal in-memory paper broker placeholder."""

from stocker_execution.broker import Broker
from stocker_execution.orders import Order, OrderSide, OrderStatus, ProposedOrder


class PaperBroker(Broker):
    """A deliberately small paper broker for future tests and dry runs."""

    def __init__(self, *, starting_cash: float = 100_000.0) -> None:
        self._cash = starting_cash
        self._positions: dict[str, float] = {}
        self._latest_prices: dict[str, float] = {}
        self._orders: dict[str, Order] = {}
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    async def get_cash(self) -> float:
        return self._cash

    async def get_latest_price(self, symbol: str) -> float:
        if symbol not in self._latest_prices:
            raise KeyError(f"No latest price for {symbol}")
        return self._latest_prices[symbol]

    def set_latest_price(self, symbol: str, price: float) -> None:
        """Set a price for paper-only tests and dry runs."""

        if price <= 0:
            raise ValueError("price must be positive")
        self._latest_prices[symbol] = price

    async def place_order(self, order: ProposedOrder) -> Order:
        if not self._connected:
            raise RuntimeError("paper broker is not connected")

        order_id = f"paper-{len(self._orders) + 1}"
        submitted = Order(**order.model_dump(), order_id=order_id, status=OrderStatus.FILLED)
        notional = order.notional
        if order.side is OrderSide.BUY:
            self._cash -= notional
            self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.quantity
        else:
            self._cash += notional
            self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) - order.quantity
        self._orders[order_id] = submitted
        return submitted

    async def cancel_order(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Unknown order id: {order_id}")
        if order.status is OrderStatus.FILLED:
            raise ValueError("filled paper orders cannot be canceled")
        self._orders[order_id] = order.model_copy(update={"status": OrderStatus.CANCELED})
