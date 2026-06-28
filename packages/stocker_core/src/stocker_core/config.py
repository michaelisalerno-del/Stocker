"""Typed configuration loading for research and execution processes."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
