"""One account, one frozen method, and explicit listed-product authorisations."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PAPER_ACCOUNT = "DUP655399"
MARKETS = ("BTC", "CL", "GC", "NG", "NQ", "SI")
RULE_VERSION = "CLOCK60_NG13_20260927"
MAX_PREMIUM_RISK_GBP = 10
MAX_OPEN_POSITIONS = 4
MAX_SIMULTANEOUS_ENTRY_RISK_GBP = 40
MAX_CONTRACTS_PER_TRADE = 1


class ProductMapping(BaseModel):
    """No default product substitution or invented delta tolerance.

    Populating this is a reviewed execution adaptation, not an arming shortcut.
    Broker metadata must independently agree on every identity/price field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    product: str
    symbol: str
    exchange: str
    trading_class: str
    currency: Literal["USD"]
    multiplier: float = Field(gt=0)
    price_unit_factor: float = Field(gt=0)
    price_magnifier: int = Field(ge=1)
    delta_tolerance: float = Field(ge=0, lt=0.1)
    expiry_timezone: str
    termination_time: str
    settlement: Literal["CASH", "FUTURES"]
    fee_reserve_gbp: float = Field(gt=0, lt=10)
    source: str = Field(min_length=10)
    approval: str = Field(min_length=10)
    strike_rule: Literal["NEAREST_FROZEN_MODEL_DELTA"]


class MarketDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    total_lines: int = Field(default=100, ge=16, le=1000)
    allowance_status: Literal["ASSUMED", "CONFIGURED", "BROKER_VERIFIED"] = "ASSUMED"
    allowance_source: str | None = None
    verified_at: str | None = None
    app_line_cap: int = Field(default=60, ge=16, le=60)
    external_headroom: int = Field(default=40, ge=0)
    known_external_lines: int | None = Field(default=None, ge=0)
    depth_slots: int = Field(default=3, ge=0, le=3)
    tick_by_tick_slots: int = Field(default=5, ge=0, le=5)
    temporary_option_quotes: int = Field(default=15, ge=1, le=15)
    option_batch_size: int = Field(default=5, ge=1, le=5)
    outbound_limit: int = Field(default=50, ge=10, le=50)
    outbound_headroom: int = Field(default=10, ge=1)
    urgent_reserve: int = Field(default=10, ge=2)
    queue_size: int = Field(default=128, ge=32, le=256)
    cancel_drain_seconds: float = Field(default=1, ge=0.1, le=5)
    rejection_backoff_seconds: int = Field(default=300, ge=60, le=3600)

    @property
    def line_budget(self) -> int:
        external = max(self.external_headroom, self.known_external_lines or 0)
        return max(0, min(self.app_line_cap, self.total_lines - external))

    @property
    def request_budget(self) -> int:
        return min(self.outbound_limit, self.total_lines // 2) - self.outbound_headroom

    @model_validator(mode="after")
    def coherent(self) -> "MarketDataConfig":
        if self.request_budget <= self.urgent_reserve:
            raise ValueError("Outbound headroom/reserve exhausts request budget")
        if self.allowance_status != "ASSUMED" and not self.allowance_source:
            raise ValueError("Configured allowance requires a source")
        if self.allowance_status == "BROKER_VERIFIED" and not self.verified_at:
            raise ValueError("Broker verification requires a timestamp")
        if self.verified_at and datetime.fromisoformat(self.verified_at).tzinfo is None:
            raise ValueError("Allowance verification timestamp requires timezone")
        return self


class L2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    levels: int = Field(default=5, ge=1, le=5)
    pre_seconds: int = Field(default=120, ge=1, le=300)
    post_seconds: int = Field(default=120, ge=1, le=300)
    dwell_seconds: int = Field(default=60, ge=10, le=300)
    stale_seconds: int = Field(default=30, ge=5, le=120)
    events_per_book: int = Field(default=10000, ge=100, le=20000)
    memory_bytes: int = Field(default=32 * 1024 * 1024, ge=65536, le=64 * 1024 * 1024)
    disk_bytes: int = Field(default=256 * 1024 * 1024, ge=65536, le=1024 * 1024 * 1024)
    writer_queue: int = Field(default=32, ge=2, le=64)


class FuturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    environment: Literal["PAPER"] = "PAPER"
    expected_account: Literal["DUP655399"] = "DUP655399"
    host: Literal["127.0.0.1"] = "127.0.0.1"
    # Existing, verified paper Gateway endpoint. No automatic port fallback.
    port: Literal[4003] = 4003
    client_id: Literal[83] = 83
    armed: bool = False
    max_premium_risk_gbp: Literal[10] = 10
    max_open_positions: Literal[4] = 4
    max_simultaneous_entry_risk_gbp: Literal[40] = 40
    max_contracts_per_trade: Literal[1] = 1
    mappings: dict[Literal["BTC", "CL", "GC", "NG", "NQ", "SI"], ProductMapping] = Field(
        default_factory=dict
    )
    quote_max_age_seconds: Literal[5] = 5
    entry_deadline_seconds: Literal[20] = 20
    exit_attempts: Literal[3] = 3
    exit_cutoff_buffer_seconds: Literal[120] = 120
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    l2: L2Config = Field(default_factory=L2Config)


def load(path: Path) -> FuturesConfig:
    return FuturesConfig.model_validate(yaml.safe_load(path.read_text()))
