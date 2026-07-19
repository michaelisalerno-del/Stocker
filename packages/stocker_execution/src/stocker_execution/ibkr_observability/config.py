"""Fail-closed local-only configuration for IBKR observability."""

from __future__ import annotations

import os
from dataclasses import dataclass

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class IBKRObserverConfig:
    """Versioned limits and connection settings with no credential fields."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 71901
    maximum_observation_delay_seconds: float = 10.0
    request_timeout_seconds: float = 10.0
    maximum_requests_per_second: int = 5
    maximum_in_flight_requests: int = 10
    bounded_retries: int = 2
    documentation_version: str = "IBKR Campus TWS API Documentation reviewed 2026-07-19"
    documentation_url: str = "https://ibkrcampus.com/campus/ibkr-api-page/twsapi-doc/"
    require_tws_read_only_api_mode: bool = True

    def __post_init__(self) -> None:
        if self.host not in _LOCAL_HOSTS:
            raise ValueError("IBKR observer permits localhost connections only")
        if not 1 <= self.port <= 65535:
            raise ValueError("invalid local TWS/IB Gateway port")
        if self.client_id < 0:
            raise ValueError("IBKR client id must be non-negative")
        if self.maximum_observation_delay_seconds != 10.0:
            raise ValueError("V1 maximum quote observation delay is frozen at ten seconds")
        if self.maximum_requests_per_second < 1 or self.maximum_in_flight_requests < 1:
            raise ValueError("request limits must be positive")
        if self.request_timeout_seconds <= 0.0:
            raise ValueError("request timeout must be positive")
        if self.bounded_retries < 0:
            raise ValueError("bounded retries must be non-negative")

    @classmethod
    def from_environment(cls) -> IBKRObserverConfig:
        """Load only non-secret local settings from explicitly named variables."""

        return cls(
            enabled=os.getenv("STOCKER_IBKR_OBSERVER_ENABLED", "false").lower() == "true",
            host=os.getenv("STOCKER_IBKR_OBSERVER_HOST", "127.0.0.1"),
            port=int(os.getenv("STOCKER_IBKR_OBSERVER_PORT", "7497")),
            client_id=int(os.getenv("STOCKER_IBKR_OBSERVER_CLIENT_ID", "71901")),
        )
