"""IBKR capability preflight and scientifically binding live-data gate."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from stocker_prospective.contract import CONTRACT_VERSION, claims_boundary
from stocker_prospective.market_data import MarketDataType


class CapabilityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connected: bool
    api_server_version: int | None
    ibkr_api_version: str | None
    tws_or_gateway_version: str | None
    market_data_type: MarketDataType | None
    underlying_level1_symbols: tuple[str, ...]
    market_proxy_level1_symbols: tuple[str, ...]
    option_level1_available: bool
    option_computation_fields_available: bool
    tick_by_tick_capacity: int | None
    depth_capacity: int | None
    option_capacity: int | None
    depth_exchanges: tuple[str, ...]
    resolved_contracts: tuple[str, ...]
    unresolved_contracts: tuple[str, ...]
    clock_drift_seconds: float | None
    new_york_calendar_valid: bool
    timestamps_valid: bool
    permission_errors: tuple[str, ...]


class IBKRCapabilityManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: str
    claims_boundary: dict[str, bool | float]
    observed_at_utc: datetime
    observation: CapabilityObservation
    required_underlyings: tuple[str, ...]
    required_market_proxies: tuple[str, ...]
    blockers: tuple[str, ...]
    scientific_recording_valid: bool
    diagnostic_display_allowed: bool
    quotes_are_live: bool
    delayed_or_frozen: bool

    @field_validator("observed_at_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("capability timestamp must be timezone-aware")
        return value.astimezone(UTC)


def run_capability_preflight(
    observation: CapabilityObservation,
    *,
    required_underlyings: tuple[str, ...],
    required_market_proxies: tuple[str, ...],
    maximum_clock_drift_seconds: float,
    output_path: str | Path | None,
    observed_at: datetime,
) -> IBKRCapabilityManifest:
    """Preserve diagnostic states while allowing science only on complete live inputs."""

    missing_underlyings = sorted(
        set(required_underlyings).difference(observation.underlying_level1_symbols)
    )
    missing_proxies = sorted(
        set(required_market_proxies).difference(observation.market_proxy_level1_symbols)
    )
    checks: list[tuple[bool, str]] = [
        (observation.connected, "ibkr_not_connected"),
        (observation.api_server_version is not None, "api_server_version_missing"),
        (
            observation.market_data_type is MarketDataType.LIVE,
            "market_data_not_live",
        ),
        (not missing_underlyings, "underlying_level1_unavailable"),
        (not missing_proxies, "market_proxy_level1_unavailable"),
        (not observation.permission_errors, "market_data_permission_error"),
        (not observation.unresolved_contracts, "contract_resolution_failed"),
        (observation.new_york_calendar_valid, "new_york_calendar_invalid"),
        (observation.timestamps_valid, "data_timestamp_invalid"),
        (
            observation.clock_drift_seconds is not None
            and abs(observation.clock_drift_seconds) <= maximum_clock_drift_seconds,
            "clock_drift_outside_tolerance",
        ),
    ]
    blockers = tuple(reason for passed, reason in checks if not passed)
    manifest = IBKRCapabilityManifest(
        contract_version=CONTRACT_VERSION,
        claims_boundary=claims_boundary(),
        observed_at_utc=observed_at,
        observation=observation,
        required_underlyings=required_underlyings,
        required_market_proxies=required_market_proxies,
        blockers=blockers,
        scientific_recording_valid=not blockers,
        diagnostic_display_allowed=observation.connected,
        quotes_are_live=observation.market_data_type is MarketDataType.LIVE,
        delayed_or_frozen=observation.market_data_type
        in {
            MarketDataType.DELAYED,
            MarketDataType.DELAYED_FROZEN,
            MarketDataType.FROZEN,
        },
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(
                manifest.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    return manifest
