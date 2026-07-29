"""Fail-closed configuration for the dedicated prospective server."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Callable
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
    raw_event_root: Path | None = None
    recorder_activation: Path | None = None
    m1c_live_parity_report: Path | None = None
    direction_live_parity_report: Path | None = None
    ibkr_capability_manifest: Path | None = None
    ibkr_runtime_capacity_manifest: Path | None = None
    prospective_report_root: Path | None = None
    aggregate_transfer_report: Path | None = None
    prospective_phase_ledger: Path | None = None
    frozen_m1c_artifact_root: Path | None = None
    m1c_scaling_artifact: Path | None = None
    m1c_tail_phase_v1_config: Path | None = None
    m1c_signed_market_shock_v1_config: Path | None = None
    m1c_opening_market_transition_v1_config: Path | None = None
    m1c_prospective_opening_reversal_v1_config: Path | None = None
    m1c_prospective_opening_reversal_v1_activation: Path | None = None
    direction_beta_artifact: Path | None = None
    historical_activity_bars: Path | None = None
    bar_compatibility_report: Path | None = None
    quiet_state_concentration_audit_root: Path | None = None


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
    expected_environment: Literal["read_only", "paper", "live_read_only"] = "read_only"
    read_only: bool = True
    market_data_type_required: Literal["live"] = "live"
    enable_level2: bool = False
    level2_rows: int = Field(default=5, ge=1, le=20)
    max_depth_subscriptions: int = Field(default=0, ge=0)
    max_tick_by_tick_subscriptions: int = Field(default=2, ge=0)
    max_option_subscriptions: int = Field(default=8, ge=0)
    externally_reserved_lines: int = Field(default=0, ge=0)
    reserved_future_trading_lines: int = Field(default=12, ge=1)
    safety_margin_lines: int = Field(default=2, ge=0)
    max_concurrent_snapshots: int = Field(default=2, ge=1)
    max_active_option_episodes: int = Field(default=1, ge=1, le=2)
    max_option_lines_per_episode: int = Field(default=8, ge=4, le=16)
    tick_by_tick_active_underlyings: int = Field(default=1, ge=0, le=2)
    level2_active_underlyings: int = Field(default=0, ge=0, le=1)
    max_high_resolution_underlyings: int = Field(default=1, ge=1, le=2)
    high_tail_approach_boundary: float = Field(default=0.40, ge=0.167095528962669, le=1.0)
    pending_subscription_timeout_seconds: float = Field(default=15.0, gt=0.0)
    subscription_reconciliation_interval_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
    )
    option_episode_maximum_minutes: int = Field(default=65, ge=30, le=90)
    connect_timeout_seconds: float = Field(default=10.0, gt=0.0)
    request_timeout_seconds: float = Field(default=10.0, gt=0.0)
    reconnect_max_attempts: int = Field(default=5, ge=0)
    reconnect_backoff_seconds: float = Field(default=2.0, gt=0.0)
    market_data_line_budget: int = Field(default=100, ge=1)
    reserved_line_headroom: int = Field(default=10, ge=0)
    request_rate_per_second: int = Field(default=20, ge=1)
    historical_requests_per_window: int = Field(default=60, ge=1)
    historical_request_window_seconds: int = Field(default=600, ge=1)
    quote_capture_timeout_seconds: float = Field(default=15.0, ge=12.0)
    allowed_market_data_types: list[MarketDataTypeName] = Field(
        default_factory=_default_market_data_types
    )
    tws_or_gateway_version: str | None = None
    maximum_clock_drift_seconds: float = Field(default=2.0, gt=0.0)
    maximum_quote_age_seconds: float = Field(default=2.0, gt=0.0)
    stream_poll_interval_seconds: float = Field(default=0.1, gt=0.0, le=1.0)
    option_strike_steps: int = Field(default=2, ge=0, le=10)
    maximum_option_contracts_per_episode: int = Field(default=30, ge=0, le=100)

    @model_validator(mode="after")
    def _headroom_is_bounded(self) -> IBKRConfig:
        try:
            host_address = ipaddress.ip_address(self.host)
        except ValueError as exc:
            raise ValueError("IBKR host must be a literal loopback address") from exc
        if not host_address.is_loopback:
            raise ValueError("IBKR host must be a literal loopback address")
        if not self.read_only:
            raise ValueError("IBKR recorder must use read-only access")
        runtime_reserved = (
            self.externally_reserved_lines
            + self.reserved_future_trading_lines
            + self.safety_margin_lines
        )
        if runtime_reserved >= self.market_data_line_budget:
            raise ValueError("runtime reservations must be below the line budget")
        usable_level1_lines = self.market_data_line_budget - runtime_reserved
        if usable_level1_lines < 21:
            raise ValueError("market-data budget cannot protect the 20 stocks and VTI")
        bounded_option_lines = min(
            self.max_option_subscriptions,
            self.max_active_option_episodes * self.max_option_lines_per_episode,
        )
        if bounded_option_lines > usable_level1_lines - 21:
            raise ValueError(
                "option subscription budget would consume protected universe Level I headroom"
            )
        if self.level2_active_underlyings > 0 and not self.enable_level2:
            raise ValueError("level2 active underlyings require enable_level2")
        if self.tick_by_tick_active_underlyings * 2 > self.max_tick_by_tick_subscriptions:
            raise ValueError("tick-by-tick active-underlying count exceeds stream capacity")
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
    credential_status_env: str = Field(
        default="STOCKER_EODHD_TOKEN_CONFIGURED",
        min_length=1,
    )
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
    parallel_validation: ParallelValidationConfig = Field(default_factory=ParallelValidationConfig)

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
    ibkr = payload.setdefault("ibkr", {})
    if not isinstance(ibkr, dict):
        raise ValueError("prospective ibkr config must be a YAML mapping")

    def environment_bool(value: str) -> bool:
        value = value.strip().lower()
        if value not in {"true", "false"}:
            raise ValueError("IBKR boolean environment values must be true or false")
        return value == "true"

    environment_fields: dict[str, tuple[str, Callable[[str], object]]] = {
        "IBKR_HOST": ("host", str),
        "IBKR_PORT": ("port", int),
        "IBKR_CLIENT_ID": ("client_id", int),
        "IBKR_READ_ONLY": ("read_only", environment_bool),
        "IBKR_MARKET_DATA_TYPE_REQUIRED": ("market_data_type_required", str),
        "IBKR_ENABLE_LEVEL2": ("enable_level2", environment_bool),
        "IBKR_LEVEL2_ROWS": ("level2_rows", int),
        "IBKR_MAX_DEPTH_SUBSCRIPTIONS": ("max_depth_subscriptions", int),
        "IBKR_MAX_DEPTH": ("max_depth_subscriptions", int),
        "IBKR_MAX_TICK_BY_TICK_SUBSCRIPTIONS": (
            "max_tick_by_tick_subscriptions",
            int,
        ),
        "IBKR_MAX_TICK_BY_TICK": ("max_tick_by_tick_subscriptions", int),
        "IBKR_MAX_OPTION_SUBSCRIPTIONS": ("max_option_subscriptions", int),
        "IBKR_TOTAL_MARKET_DATA_LINES": ("market_data_line_budget", int),
        "IBKR_EXTERNALLY_RESERVED_LINES": ("externally_reserved_lines", int),
        "IBKR_RESERVED_FUTURE_TRADING_LINES": (
            "reserved_future_trading_lines",
            int,
        ),
        "IBKR_MAX_CONCURRENT_SNAPSHOTS": ("max_concurrent_snapshots", int),
        "IBKR_MAX_ACTIVE_OPTION_EPISODES": ("max_active_option_episodes", int),
        "IBKR_MAX_OPTION_LINES_PER_EPISODE": (
            "max_option_lines_per_episode",
            int,
        ),
        "IBKR_HISTORICAL_REQUESTS_PER_WINDOW": (
            "historical_requests_per_window",
            int,
        ),
        "IBKR_HISTORICAL_REQUEST_WINDOW_SECONDS": (
            "historical_request_window_seconds",
            int,
        ),
        "IBKR_CONNECTION_TIMEOUT_SECONDS": ("connect_timeout_seconds", float),
        "IBKR_RECONNECT_BACKOFF_SECONDS": ("reconnect_backoff_seconds", float),
        "IBKR_STREAM_POLL_INTERVAL_SECONDS": ("stream_poll_interval_seconds", float),
        "IBKR_TWS_OR_GATEWAY_VERSION": ("tws_or_gateway_version", str),
    }
    for environment_name, (field_name, converter) in environment_fields.items():
        if environment_name not in os.environ:
            continue
        ibkr[field_name] = converter(os.environ[environment_name])
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
    if not config.ibkr.read_only:
        reasons.append("IBKR read-only access is required")
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
            "externally_reserved_lines": config.ibkr.externally_reserved_lines,
            "reserved_future_trading_lines": (config.ibkr.reserved_future_trading_lines),
            "safety_margin_lines": config.ibkr.safety_margin_lines,
            "allowed_market_data_types": config.ibkr.allowed_market_data_types,
            "read_only": config.ibkr.read_only,
            "market_data_type_required": config.ibkr.market_data_type_required,
            "enable_level2": config.ibkr.enable_level2,
            "level2_rows": config.ibkr.level2_rows,
            "max_depth_subscriptions": config.ibkr.max_depth_subscriptions,
            "max_tick_by_tick_subscriptions": (config.ibkr.max_tick_by_tick_subscriptions),
            "max_option_subscriptions": config.ibkr.max_option_subscriptions,
            "max_active_option_episodes": config.ibkr.max_active_option_episodes,
            "max_option_lines_per_episode": (config.ibkr.max_option_lines_per_episode),
            "max_concurrent_snapshots": config.ibkr.max_concurrent_snapshots,
            "historical_requests_per_window": (config.ibkr.historical_requests_per_window),
            "historical_request_window_seconds": (config.ibkr.historical_request_window_seconds),
            "tick_by_tick_active_underlyings": (config.ibkr.tick_by_tick_active_underlyings),
            "level2_active_underlyings": config.ibkr.level2_active_underlyings,
        },
        "context": {"mode": config.context.mode},
        "parallel_validation": {
            "enabled": config.parallel_validation.enabled,
            "provider": config.parallel_validation.provider,
            "credential_configured": (
                os.environ.get(config.parallel_validation.credential_status_env) == "1"
            ),
            "capture_delay_seconds": config.parallel_validation.capture_delay_seconds,
        },
    }
