"""Fail-closed configuration for the dedicated prospective server."""

from __future__ import annotations

import ipaddress
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RuntimeSafetyError(RuntimeError):
    """Unsafe prospective runtime configuration."""


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database: Path
    bundle_root: Path
    feature_parity_report: Path
    context_root: Path | None = None
    replay_universe: Path | None = None


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["record_only", "shadow"]
    source: Literal["replay", "ibkr"]
    prospective_start_utc: datetime
    instance_id: str = Field(min_length=1)
    app_version: str = Field(min_length=1)
    git_commit: str = Field(pattern=r"^[a-f0-9]{7,64}$")
    run_id: str | None = None
    recorder_lease_stale_seconds: int = Field(default=60, ge=15)
    heartbeat_seconds: int = Field(default=10, ge=1)

    @field_validator("prospective_start_utc")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("prospective_start_utc must be timezone-aware")
        return value


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trading_enabled: bool = False


class WebConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65_535)
    production: bool = True
    trust_proxy_headers: bool = False
    trusted_proxy_ips: list[str] = Field(default_factory=list)
    authentication_enabled: bool = False
    auth_token_env: str | None = None
    auth_cookie_name: str = "__Host-stocker_session"
    auth_cookie_secure: bool = True
    requests_per_minute: int = Field(default=120, ge=1, le=10_000)
    allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost"])

    @model_validator(mode="after")
    def _auth_has_environment_name(self) -> WebConfig:
        if self.host in {"0.0.0.0", "::"}:
            raise ValueError("web host may not bind all interfaces")
        if self.authentication_enabled and not self.auth_token_env:
            raise ValueError("auth_token_env is required when authentication is enabled")
        if self.authentication_enabled and self.production and not self.auth_cookie_secure:
            raise ValueError("production authentication requires secure cookies")
        if self.authentication_enabled and not self.auth_cookie_name.startswith("__Host-"):
            raise ValueError("authentication cookie must use the __Host- prefix")
        if self.trust_proxy_headers and not self.trusted_proxy_ips:
            raise ValueError("trusted_proxy_ips are required before proxy headers are trusted")
        return self


MarketDataTypeName = Literal["live", "frozen", "delayed", "delayed_frozen"]


def _default_market_data_types() -> list[MarketDataTypeName]:
    return ["live"]


class IBKRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int | None = Field(default=None, ge=1, le=65_535)
    client_id: int = Field(default=71, ge=1)
    expected_environment: Literal["paper"] = "paper"
    connect_timeout_seconds: float = Field(default=10.0, gt=0.0)
    request_timeout_seconds: float = Field(default=10.0, gt=0.0)
    reconnect_max_attempts: int = Field(default=5, ge=0)
    reconnect_backoff_seconds: float = Field(default=2.0, gt=0.0)
    market_data_line_budget: int = Field(default=100, ge=1)
    reserved_line_headroom: int = Field(default=10, ge=0)
    request_rate_per_second: int = Field(default=20, ge=1)
    quote_capture_timeout_seconds: float = Field(default=15.0, ge=12.0)
    allowed_market_data_types: list[MarketDataTypeName] = Field(
        default_factory=_default_market_data_types
    )

    @model_validator(mode="after")
    def _headroom_is_bounded(self) -> IBKRConfig:
        try:
            host_address = ipaddress.ip_address(self.host)
        except ValueError as exc:
            raise ValueError("IBKR host must be a literal loopback address") from exc
        if not host_address.is_loopback:
            raise ValueError("IBKR host must be a literal loopback address")
        if self.reserved_line_headroom >= self.market_data_line_budget:
            raise ValueError("reserved line headroom must be below the line budget")
        if self.request_rate_per_second > self.market_data_line_budget / 2:
            raise ValueError("request-rate budget must not exceed half the configured line budget")
        return self


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["signed_import"]
    hmac_secret_env: str = Field(min_length=1)
    import_directory: Path | None = None


class ParallelValidationConfig(BaseModel):
    """Bounded vendor capture used only to establish source-feature parity."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: Literal["eodhd"] = "eodhd"
    api_token_env: str = Field(default="EODHD_API_TOKEN", min_length=1)
    base_url: Literal["https://eodhd.com/api"] = "https://eodhd.com/api"
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=1, le=10)
    capture_delay_seconds: int = Field(default=7200, ge=60, le=43_200)
    requests_per_minute: int = Field(default=20, ge=1, le=60)


class ProspectiveConfig(BaseModel):
    """Top-level prospective recorder/web configuration."""

    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig
    runtime: RuntimeConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    ibkr: IBKRConfig = Field(default_factory=IBKRConfig)
    context: ContextConfig
    parallel_validation: ParallelValidationConfig = Field(
        default_factory=ParallelValidationConfig
    )

    @model_validator(mode="after")
    def _ibkr_port_is_explicit(self) -> ProspectiveConfig:
        if self.runtime.source == "ibkr" and self.ibkr.port is None:
            raise ValueError("IBKR port must be explicitly configured")
        if self.runtime.heartbeat_seconds * 2 >= self.runtime.recorder_lease_stale_seconds:
            raise ValueError("recorder heartbeat must be less than half the stale-lease interval")
        longest_blocking_call = max(
            self.ibkr.connect_timeout_seconds,
            self.ibkr.request_timeout_seconds,
            self.ibkr.quote_capture_timeout_seconds,
        )
        if (
            self.runtime.source == "ibkr"
            and longest_blocking_call + self.runtime.heartbeat_seconds
            >= self.runtime.recorder_lease_stale_seconds
        ):
            raise ValueError(
                "IBKR timeout plus heartbeat must remain below the stale-lease interval"
            )
        return self


ORDER_METHOD_NAMES = frozenset(
    {
        "cancel_order",
        "place_order",
        "submit_order",
        "transmit_order",
        "placeOrder",
        "cancelOrder",
        "reqGlobalCancel",
        "exerciseOptions",
    }
)


def load_prospective_config(path: str | Path) -> ProspectiveConfig:
    """Load strict prospective YAML without accepting unknown fields."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("prospective config must be a YAML mapping")
    runtime = payload.get("runtime")
    if isinstance(runtime, dict) and runtime.get("git_commit") == "${STOCKER_GIT_COMMIT}":
        commit = os.environ.get("STOCKER_GIT_COMMIT")
        if not commit:
            raise RuntimeSafetyError(
                "blocked_unsafe_runtime_configuration: STOCKER_GIT_COMMIT is absent"
            )
        runtime["git_commit"] = commit
    return ProspectiveConfig.model_validate(payload)


def validate_runtime_safety(config: ProspectiveConfig, market_data_adapter: object) -> None:
    """Reject trading, live mode, missing start, and order-capable adapter paths."""

    reasons: list[str] = []
    if config.risk.trading_enabled:
        reasons.append("risk.trading_enabled must be false")
    if config.runtime.mode not in {"record_only", "shadow"}:
        reasons.append("recorder mode must be record_only or shadow")
    if config.runtime.prospective_start_utc is None:
        reasons.append("prospective_start_utc is required")
    exposed = sorted(name for name in ORDER_METHOD_NAMES if hasattr(market_data_adapter, name))
    if exposed:
        reasons.append(f"order-capable runtime path exposes {', '.join(exposed)}")
    if reasons:
        raise RuntimeSafetyError("blocked_unsafe_runtime_configuration: " + "; ".join(reasons))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_persistent_paths(config: ProspectiveConfig, release_directory: str | Path) -> None:
    """Ensure rollbacks cannot replace the database or installed bundle store."""

    release = Path(release_directory)
    unsafe = [
        name
        for name, path in (
            ("database", config.paths.database),
            ("bundle_root", config.paths.bundle_root),
        )
        if _is_within(path, release)
    ]
    if unsafe:
        raise RuntimeSafetyError(
            "blocked_unsafe_runtime_configuration: database and bundles must remain "
            f"outside the application release directory ({', '.join(unsafe)})"
        )


def public_config(config: ProspectiveConfig) -> dict[str, object]:
    """Return a secret-free public configuration projection."""

    return {
        "runtime": {
            "mode": config.runtime.mode,
            "source": config.runtime.source,
            "instance_id": config.runtime.instance_id,
            "app_version": config.runtime.app_version,
            "git_commit": config.runtime.git_commit,
            "prospective_start_utc": config.runtime.prospective_start_utc.isoformat(),
        },
        "web": {
            "host": config.web.host,
            "port": config.web.port,
            "production": config.web.production,
            "authentication_enabled": config.web.authentication_enabled,
            "trust_proxy_headers": config.web.trust_proxy_headers,
        },
        "safety": {
            "trading_enabled": config.risk.trading_enabled,
            "order_path": "absent",
        },
        "ibkr": {
            "host": config.ibkr.host,
            "port_configured": config.ibkr.port is not None,
            "expected_environment": config.ibkr.expected_environment,
            "market_data_line_budget": config.ibkr.market_data_line_budget,
            "reserved_line_headroom": config.ibkr.reserved_line_headroom,
            "allowed_market_data_types": config.ibkr.allowed_market_data_types,
        },
        "context": {"mode": config.context.mode},
        "parallel_validation": {
            "enabled": config.parallel_validation.enabled,
            "provider": config.parallel_validation.provider,
            "credential_configured": bool(
                os.environ.get(config.parallel_validation.api_token_env)
            ),
            "capture_delay_seconds": config.parallel_validation.capture_delay_seconds,
        },
    }
