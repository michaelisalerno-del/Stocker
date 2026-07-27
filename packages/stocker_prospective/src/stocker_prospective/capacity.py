"""Runtime-discovered IBKR market-data capacity with explicit provenance."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from stocker_prospective.contract import claims_boundary

CapacityScalar = bool | int | str


@dataclass(frozen=True)
class CapacityValue:
    """One capacity fact and the source used to resolve it."""

    value: CapacityScalar
    source: str
    discovered: bool
    environment_variable: str | None = None


@dataclass(frozen=True)
class RuntimeCapacitySettings:
    """Safe configured fallbacks for limits IBKR cannot expose."""

    configured_total_market_data_lines: int = 100
    configured_externally_reserved_lines: int = 0
    reserved_future_trading_lines: int = 12
    safety_margin_lines: int = 2
    configured_max_tick_by_tick: int = 1
    configured_max_depth: int = 0
    configured_max_concurrent_snapshots: int = 2
    configured_max_active_option_episodes: int = 1
    configured_max_option_lines_per_episode: int = 8
    configured_historical_requests_per_window: int = 60
    configured_option_computation_available: bool = False

    def __post_init__(self) -> None:
        nonnegative = (
            self.configured_externally_reserved_lines,
            self.reserved_future_trading_lines,
            self.safety_margin_lines,
            self.configured_max_tick_by_tick,
            self.configured_max_depth,
            self.configured_max_concurrent_snapshots,
            self.configured_max_active_option_episodes,
            self.configured_max_option_lines_per_episode,
            self.configured_historical_requests_per_window,
        )
        if self.configured_total_market_data_lines <= 0:
            raise ValueError("configured total market-data lines must be positive")
        if any(value < 0 for value in nonnegative):
            raise ValueError("configured capacity values must be nonnegative")
        if self.configured_max_active_option_episodes not in {1, 2}:
            raise ValueError("configured active option episodes must be one or two")
        if self.configured_max_option_lines_per_episode < 4:
            raise ValueError("configured option lines must secure the four primary legs")


@dataclass(frozen=True)
class CapacityDiscovery:
    """Facts observable from IBKR callbacks, TWS, or the owning API client."""

    total_level1_allowance: int | None = None
    available_level1_capacity: int | None = None
    externally_consumed_lines: int | None = None
    tws_watchlist_lines: int | None = None
    other_api_client_lines: int | None = None
    current_internal_level1_lines: int = 0
    tick_by_tick_capacity: int | None = None
    tick_by_tick_in_use: int = 0
    depth_capacity: int | None = None
    depth_in_use: int = 0
    snapshot_pacing_limit: int | None = None
    historical_requests_per_window: int | None = None
    option_computation_available: bool | None = None
    market_data_status: str = "unknown"

    def __post_init__(self) -> None:
        numeric_values = (
            self.total_level1_allowance,
            self.available_level1_capacity,
            self.externally_consumed_lines,
            self.tws_watchlist_lines,
            self.other_api_client_lines,
            self.current_internal_level1_lines,
            self.tick_by_tick_capacity,
            self.tick_by_tick_in_use,
            self.depth_capacity,
            self.depth_in_use,
            self.snapshot_pacing_limit,
            self.historical_requests_per_window,
        )
        if any(value is not None and value < 0 for value in numeric_values):
            raise ValueError("discovered capacity values must be nonnegative")


@dataclass(frozen=True)
class RuntimeCapacityManifest:
    """Machine-readable startup capacity contract written before subscriptions."""

    observed_at_utc: datetime
    claims_boundary: dict[str, bool | float | str]
    total_level1_allowance: CapacityValue
    externally_reserved_lines: CapacityValue
    tws_watchlist_lines: CapacityValue
    other_api_client_lines: CapacityValue
    current_internal_level1_lines: int
    reserved_future_trading_lines: CapacityValue
    safety_margin_lines: CapacityValue
    available_ordinary_level1_lines: int
    available_research_level1_lines: int
    tick_by_tick_capacity: CapacityValue
    tick_by_tick_in_use: int
    available_tick_by_tick: int
    depth_capacity: CapacityValue
    depth_in_use: int
    available_depth: int
    snapshot_pacing_limit: CapacityValue
    max_active_option_episodes: CapacityValue
    max_option_lines_per_episode: CapacityValue
    historical_requests_per_window: CapacityValue
    option_computation_available: CapacityValue
    market_data_status: CapacityValue

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["observed_at_utc"] = self.observed_at_utc.astimezone(UTC).isoformat()
        return payload

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


def _environment_integer(
    environment: Mapping[str, str],
    name: str,
    fallback: int,
) -> tuple[int, str, str | None]:
    raw = environment.get(name)
    if raw is None:
        return fallback, "configured_fallback", None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value, "configured_environment", name


def _resolved_integer(
    *,
    discovered: int | None,
    environment: Mapping[str, str],
    environment_name: str,
    fallback: int,
) -> CapacityValue:
    if discovered is not None:
        return CapacityValue(discovered, "ibkr_discovery", True)
    value, source, variable = _environment_integer(environment, environment_name, fallback)
    return CapacityValue(value, source, False, variable)


def resolve_runtime_capacity(
    *,
    settings: RuntimeCapacitySettings,
    discovery: CapacityDiscovery,
    environment: Mapping[str, str] | None = None,
    observed_at: datetime | None = None,
    output_path: str | Path | None = None,
) -> RuntimeCapacityManifest:
    """Resolve discovered values first and configured fallbacks second."""

    values = os.environ if environment is None else environment
    observed = datetime.now(UTC) if observed_at is None else observed_at.astimezone(UTC)
    total = _resolved_integer(
        discovered=discovery.total_level1_allowance,
        environment=values,
        environment_name="IBKR_TOTAL_MARKET_DATA_LINES",
        fallback=settings.configured_total_market_data_lines,
    )
    tws = CapacityValue(
        discovery.tws_watchlist_lines or 0,
        "ibkr_discovery" if discovery.tws_watchlist_lines is not None else "not_observable",
        discovery.tws_watchlist_lines is not None,
    )
    other_clients = CapacityValue(
        discovery.other_api_client_lines or 0,
        "ibkr_discovery" if discovery.other_api_client_lines is not None else "not_observable",
        discovery.other_api_client_lines is not None,
    )
    discovered_external = discovery.externally_consumed_lines
    if discovered_external is None and (
        discovery.tws_watchlist_lines is not None or discovery.other_api_client_lines is not None
    ):
        discovered_external = int(tws.value) + int(other_clients.value)
    external = _resolved_integer(
        discovered=discovered_external,
        environment=values,
        environment_name="IBKR_EXTERNALLY_RESERVED_LINES",
        fallback=settings.configured_externally_reserved_lines,
    )
    total_value = int(total.value)
    if discovery.available_level1_capacity is not None:
        implied_external = max(
            0,
            total_value - discovery.available_level1_capacity,
        )
        if implied_external > int(external.value):
            external = CapacityValue(
                implied_external,
                "ibkr_discovery_available_capacity",
                True,
            )
    future = _resolved_integer(
        discovered=None,
        environment=values,
        environment_name="IBKR_RESERVED_FUTURE_TRADING_LINES",
        fallback=settings.reserved_future_trading_lines,
    )
    safety = CapacityValue(
        settings.safety_margin_lines,
        "configured_fallback",
        False,
    )
    tick = _resolved_integer(
        discovered=discovery.tick_by_tick_capacity,
        environment=values,
        environment_name="IBKR_MAX_TICK_BY_TICK",
        fallback=settings.configured_max_tick_by_tick,
    )
    depth = _resolved_integer(
        discovered=discovery.depth_capacity,
        environment=values,
        environment_name="IBKR_MAX_DEPTH",
        fallback=settings.configured_max_depth,
    )
    snapshots = _resolved_integer(
        discovered=discovery.snapshot_pacing_limit,
        environment=values,
        environment_name="IBKR_MAX_CONCURRENT_SNAPSHOTS",
        fallback=settings.configured_max_concurrent_snapshots,
    )
    active_option_episodes = _resolved_integer(
        discovered=None,
        environment=values,
        environment_name="IBKR_MAX_ACTIVE_OPTION_EPISODES",
        fallback=settings.configured_max_active_option_episodes,
    )
    option_lines_per_episode = _resolved_integer(
        discovered=None,
        environment=values,
        environment_name="IBKR_MAX_OPTION_LINES_PER_EPISODE",
        fallback=settings.configured_max_option_lines_per_episode,
    )
    if int(active_option_episodes.value) not in {1, 2}:
        raise ValueError("IBKR_MAX_ACTIVE_OPTION_EPISODES must be one or two")
    if int(option_lines_per_episode.value) < 4:
        raise ValueError("IBKR_MAX_OPTION_LINES_PER_EPISODE must be at least four")
    historical = _resolved_integer(
        discovered=discovery.historical_requests_per_window,
        environment=values,
        environment_name="IBKR_HISTORICAL_REQUESTS_PER_WINDOW",
        fallback=settings.configured_historical_requests_per_window,
    )
    option_computation = CapacityValue(
        (
            discovery.option_computation_available
            if discovery.option_computation_available is not None
            else settings.configured_option_computation_available
        ),
        (
            "ibkr_discovery"
            if discovery.option_computation_available is not None
            else "configured_fallback"
        ),
        discovery.option_computation_available is not None,
    )
    externally_used = int(external.value)
    available_ordinary = max(
        0,
        min(
            total_value - externally_used,
            (
                discovery.available_level1_capacity
                if discovery.available_level1_capacity is not None
                else total_value - externally_used
            ),
        ),
    )
    available_research = max(
        0,
        available_ordinary
        - int(future.value)
        - int(safety.value)
        - discovery.current_internal_level1_lines,
    )
    manifest = RuntimeCapacityManifest(
        observed_at_utc=observed,
        claims_boundary=claims_boundary(),
        total_level1_allowance=total,
        externally_reserved_lines=external,
        tws_watchlist_lines=tws,
        other_api_client_lines=other_clients,
        current_internal_level1_lines=discovery.current_internal_level1_lines,
        reserved_future_trading_lines=future,
        safety_margin_lines=safety,
        available_ordinary_level1_lines=available_ordinary,
        available_research_level1_lines=available_research,
        tick_by_tick_capacity=tick,
        tick_by_tick_in_use=discovery.tick_by_tick_in_use,
        available_tick_by_tick=max(0, int(tick.value) - discovery.tick_by_tick_in_use),
        depth_capacity=depth,
        depth_in_use=discovery.depth_in_use,
        available_depth=max(0, int(depth.value) - discovery.depth_in_use),
        snapshot_pacing_limit=snapshots,
        max_active_option_episodes=active_option_episodes,
        max_option_lines_per_episode=option_lines_per_episode,
        historical_requests_per_window=historical,
        option_computation_available=option_computation,
        market_data_status=CapacityValue(
            discovery.market_data_status,
            "ibkr_discovery",
            True,
        ),
    )
    if output_path is not None:
        manifest.write(output_path)
    return manifest


__all__ = [
    "CapacityDiscovery",
    "CapacityValue",
    "RuntimeCapacityManifest",
    "RuntimeCapacitySettings",
    "resolve_runtime_capacity",
]
