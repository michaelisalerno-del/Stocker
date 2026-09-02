"""Typed configuration loading for research and execution processes."""

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, StringConstraints, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from stocker_core.runs import Environment, RunConfig
from stocker_core.universes import (
    NAMED_US_UNIVERSES,
    UniverseDefinition,
    load_us_universe_snapshot,
)


class DataConfig(BaseModel):
    """Filesystem and market-context settings shared by research and execution."""

    data_dir: Path = Path("data")
    timezone: str = "UTC"
    default_currency: str = "USD"


class EODHDConfig(BaseModel):
    """EODHD data-vendor settings without secrets."""

    enabled: bool = False
    base_url: str = "https://eodhd.com/api"
    api_token_env: str = "EODHD_API_TOKEN"
    default_fmt: Literal["json"] = "json"
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=1)
    save_raw_by_default: bool = True


class DataVendorsConfig(BaseModel):
    """Optional data-vendor configs used only by the data pipeline."""

    eodhd: EODHDConfig = Field(default_factory=EODHDConfig)


class CostsConfig(BaseModel):
    """Basic transaction-cost assumptions in basis points."""

    spread_bps: float = Field(default=0.0, ge=0.0)
    commission_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)

    def one_way_bps(self) -> float:
        """Return the estimated one-way cost in basis points."""

        return self.spread_bps + self.commission_bps + self.slippage_bps

    def round_trip_bps(self) -> float:
        """Return the estimated entry-plus-exit cost in basis points."""

        return self.one_way_bps() * 2


class RiskConfig(BaseModel):
    """Hard risk limits used before any order can be considered."""

    max_position_size: float = Field(default=0.0, ge=0.0)
    max_order_size: float = Field(default=0.0, ge=0.0)
    max_daily_loss: float = Field(default=0.0, ge=0.0)
    max_orders_per_day: int = Field(default=0, ge=0)
    trading_enabled: bool = False


class ResearchSettings(BaseModel):
    """Research-only settings that should never be required by the server."""

    starting_cash: float = Field(default=100_000.0, gt=0.0)
    benchmark_symbol: str | None = None


class BrokerConfig(BaseModel):
    """Placeholder broker configuration without credentials."""

    provider: str = "placeholder"
    account_id_env: str | None = None
    api_key_env: str | None = None


class IbkrConfig(BaseModel):
    """Explicit connection settings for one IBKR PAPER or LIVE session."""

    environment: Environment
    host: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    port: int = Field(ge=1, le=65_535)
    client_id: int = Field(ge=1)
    expected_account: str | None = Field(default=None, min_length=1)
    connect_timeout_seconds: float = Field(default=5.0, gt=0.0)
    request_timeout_seconds: float = Field(default=60.0, gt=0.0)


class RunsConfig(BaseModel):
    """Configured universes and the independent runs that reference them."""

    universes: tuple[UniverseDefinition, ...] = Field(min_length=1)
    runs: tuple[RunConfig, ...] = ()

    @model_validator(mode="after")
    def validate_run_references(self) -> "RunsConfig":
        """Reject duplicate identities and runs that name an absent universe."""

        universe_ids: set[str] = set()
        universes_by_id: dict[str, UniverseDefinition] = {}
        for universe in self.universes:
            if universe.universe_id in universe_ids:
                raise ValueError(f"Duplicate universe_id: {universe.universe_id}")
            universe_ids.add(universe.universe_id)
            universes_by_id[universe.universe_id] = universe

        run_ids: set[str] = set()
        for run in self.runs:
            if run.run_id in run_ids:
                raise ValueError(f"Duplicate run_id: {run.run_id}")
            run_ids.add(run.run_id)
            if run.universe not in universe_ids:
                raise ValueError(f"Unknown universe {run.universe} referenced by run {run.run_id}")
            if run.market_id is not None:
                market_spec = universes_by_id[run.universe].market_spec
                if market_spec is None or (
                    market_spec.market_id is not run.market_id
                    or market_spec.cap_bucket is not run.cap_bucket
                    or market_spec.cap_bucket_version != run.cap_bucket_version
                ):
                    raise ValueError(
                        f"Run {run.run_id} lineage does not match universe {run.universe}"
                    )
        return self


class ServerSettings(BaseModel):
    """Server runtime settings for future paper/live execution."""

    mode: Literal["paper", "live"] = "paper"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65_535)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)


class ResearchConfig(BaseSettings):
    """Top-level config for Mac research and backtesting workflows."""

    model_config = SettingsConfigDict(
        env_prefix="STOCKER_", env_nested_delimiter="__", extra="ignore"
    )

    data: DataConfig = Field(default_factory=DataConfig)
    data_vendors: DataVendorsConfig = Field(default_factory=DataVendorsConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    research: ResearchSettings = Field(default_factory=ResearchSettings)


class ServerConfig(BaseSettings):
    """Top-level config for server-side dry-run, paper, and future live execution."""

    model_config = SettingsConfigDict(
        env_prefix="STOCKER_", env_nested_delimiter="__", extra="ignore"
    )

    data: DataConfig = Field(default_factory=DataConfig)
    data_vendors: DataVendorsConfig = Field(default_factory=DataVendorsConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    server: ServerSettings = Field(default_factory=ServerSettings)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return raw


def load_config[ConfigT: BaseModel](path: str | Path, config_type: type[ConfigT]) -> ConfigT:
    """Load a typed config model from a YAML file."""

    return config_type.model_validate(_read_yaml(path))


def load_research_config(path: str | Path) -> ResearchConfig:
    """Load a research config YAML file."""

    return load_config(path, ResearchConfig)


def load_server_config(path: str | Path) -> ServerConfig:
    """Load a server config YAML file."""

    return load_config(path, ServerConfig)


def load_run_config(path: str | Path) -> RunConfig:
    """Load a Stage 1 run config YAML file."""

    return load_config(path, RunConfig)


def load_runs_config(path: str | Path) -> RunsConfig:
    """Load the broker-independent multiple-universe and multiple-run configuration."""

    config_path = Path(path)
    raw = _read_yaml(config_path)
    snapshot_value = raw.pop("named_universe_snapshot", None)
    inline = raw.get("universes", [])
    runs = raw.get("runs", [])
    if not isinstance(inline, list) or not isinstance(runs, list):
        return RunsConfig.model_validate(raw)
    inline_ids = {str(item.get("universe_id", "")) for item in inline if isinstance(item, dict)}
    referenced = tuple(
        dict.fromkeys(
            str(item.get("universe", ""))
            for item in runs
            if isinstance(item, dict) and item.get("universe")
        )
    )
    requested_named = tuple(NAMED_US_UNIVERSES) if snapshot_value is not None else referenced
    missing_named = tuple(
        universe_id
        for universe_id in requested_named
        if universe_id in NAMED_US_UNIVERSES and universe_id not in inline_ids
    )
    if missing_named:
        if not isinstance(snapshot_value, str) or not snapshot_value.strip():
            names = ", ".join(missing_named)
            raise ValueError(
                f"Named universe {names} requires named_universe_snapshot in {config_path}"
            )
        snapshot_path = Path(snapshot_value)
        if not snapshot_path.is_absolute():
            snapshot_path = config_path.parent / snapshot_path
        snapshot = load_us_universe_snapshot(snapshot_path)
        raw["universes"] = [
            *inline,
            *(
                snapshot.get_universe(universe_id).model_dump(mode="python")
                for universe_id in missing_named
            ),
        ]
    return RunsConfig.model_validate(raw)


def load_ibkr_config(path: str | Path, environment: Environment) -> IbkrConfig:
    """Load the explicit IBKR settings selected by a Stage 1 run environment."""

    raw = _read_yaml(path)
    selected = raw.get(environment.value)
    if not isinstance(selected, dict):
        raise ValueError(f"IBKR config has no {environment.value} mapping: {path}")
    config = IbkrConfig.model_validate(selected)
    if config.environment is not environment:
        raise ValueError(
            f"IBKR {environment.value} mapping declares environment {config.environment.value}"
        )
    return config
