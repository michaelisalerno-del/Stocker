"""Order models shared by risk checks and broker interfaces."""

from enum import StrEnum

from pydantic import BaseModel, Field


class OrderSide(StrEnum):
    """Supported order sides."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Supported placeholder order types."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    """Simple order lifecycle states."""

    PENDING = "pending"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class ProposedOrder(BaseModel):
    """A requested order before risk approval."""

    symbol: str = Field(min_length=1)
    side: OrderSide
    quantity: float = Field(gt=0.0)
    estimated_price: float = Field(gt=0.0)
    order_type: OrderType = OrderType.MARKET

    @property
    def notional(self) -> float:
        """Return estimated cash notional."""

        return self.quantity * self.estimated_price


class Order(ProposedOrder):
    """A paper/live order record after submission."""

    order_id: str
    status: OrderStatus = OrderStatus.PENDING
