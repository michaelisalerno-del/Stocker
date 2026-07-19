"""Minimal client protocol that cannot express orders, positions, or accounts."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from stocker_execution.ibkr_observability.models import (
    ContractIdentity,
    ContractRequest,
    QuoteSnapshot,
)


class IBKRObservabilityClient(Protocol):
    """Only operations required for current contract and top-of-book observation."""

    @property
    def api_tws_version(self) -> str | None:
        """Return the connected API/TWS version when known."""

    async def connect(self, *, host: str, port: int, client_id: int) -> None:
        """Connect to a local TWS or IB Gateway socket."""

    async def disconnect(self) -> None:
        """Disconnect the observability client."""

    async def request_server_time(self) -> datetime:
        """Request current IBKR server time."""

    async def resolve_stock_contract(self, request: ContractRequest) -> ContractIdentity:
        """Resolve a stable stock contract identity."""

    async def capture_top_of_book_snapshot(
        self,
        *,
        request_id: int,
        contract: ContractIdentity,
        timeout_seconds: float,
    ) -> QuoteSnapshot:
        """Capture one top-of-book snapshot callback aggregation."""

    async def cancel_market_data(self, request_id: int) -> None:
        """Cancel only the market-data subscription created by this observer."""
