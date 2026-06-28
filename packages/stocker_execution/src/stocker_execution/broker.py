"""Abstract broker interface.

Real broker implementations must live behind this boundary and must not bypass risk checks.
"""

from abc import ABC, abstractmethod

from stocker_execution.orders import Order, ProposedOrder


class Broker(ABC):
    """Broker interface for future paper and live adapters."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the broker or paper environment."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect cleanly."""

    @abstractmethod
    async def get_positions(self) -> dict[str, float]:
        """Return current positions."""

    @abstractmethod
    async def get_cash(self) -> float:
        """Return available cash."""

    @abstractmethod
    async def get_latest_price(self, symbol: str) -> float:
        """Return the latest known price for a symbol."""

    @abstractmethod
    async def place_order(self, order: ProposedOrder) -> Order:
        """Submit an order after external risk checks pass."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
