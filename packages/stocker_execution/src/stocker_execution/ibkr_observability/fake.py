"""Complete deterministic fake client for offline tests and dry runs."""

from __future__ import annotations

from datetime import UTC, datetime

from stocker_execution.ibkr_observability.models import (
    ContractIdentity,
    ContractRequest,
    QuoteSnapshot,
)


class FakeIBKRObservabilityClient:
    """In-memory client that never opens a network connection."""

    def __init__(
        self,
        *,
        contracts: dict[str, ContractIdentity] | None = None,
        snapshots: dict[int, QuoteSnapshot] | None = None,
        server_time: datetime | None = None,
    ) -> None:
        self.contracts = contracts or {}
        self.snapshots = snapshots or {}
        self.server_time = server_time or datetime(2025, 7, 7, 14, 0, tzinfo=UTC)
        self.connected = False
        self.connection_attempts = 0
        self.capture_attempts: list[int] = []
        self.cancelled_request_ids: list[int] = []
        self.account_requests = 0
        self.order_requests = 0

    @property
    def api_tws_version(self) -> str:
        return "fake-tws-api-v1"

    async def connect(self, *, host: str, port: int, client_id: int) -> None:
        del port, client_id
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("fake preserves local-only connection contract")
        self.connection_attempts += 1
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def request_server_time(self) -> datetime:
        if not self.connected:
            raise RuntimeError("fake client is disconnected")
        return self.server_time

    async def resolve_stock_contract(self, request: ContractRequest) -> ContractIdentity:
        if not self.connected:
            raise RuntimeError("fake client is disconnected")
        if request.research_symbol not in self.contracts:
            raise LookupError(f"no fake contract for {request.research_symbol}")
        return self.contracts[request.research_symbol]

    async def capture_top_of_book_snapshot(
        self,
        *,
        request_id: int,
        contract: ContractIdentity,
        timeout_seconds: float,
    ) -> QuoteSnapshot:
        del contract, timeout_seconds
        self.capture_attempts.append(request_id)
        if not self.connected:
            raise RuntimeError("fake client is disconnected")
        if request_id not in self.snapshots:
            raise TimeoutError(f"no fake snapshot for request {request_id}")
        return self.snapshots[request_id]

    async def cancel_market_data(self, request_id: int) -> None:
        self.cancelled_request_ids.append(request_id)
