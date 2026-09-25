"""One PAPER account and the explicit research-to-listed-contract choices."""

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PAPER_ACCOUNT = "DUP655399"


class OrderFlowConfig(BaseModel):
    """Optional evidence only; budgets are operator-confirmed spare entitlement."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = False
    order_authoritative: Literal[False] = False
    may_submit_orders: Literal[False] = False
    max_stocks: int = Field(default=4, ge=0, le=4)
    feed_mode: Literal["TBT_TRADES_TBT_QUOTES", "TBT_TRADES_L1_QUOTES"] = "TBT_TRADES_TBT_QUOTES"
    available_tbt: int = Field(default=0, ge=0, le=1000)
    reserved_tbt: int = Field(default=4, ge=4, le=1000)
    available_l1: int = Field(default=0, ge=0, le=10000)
    reserved_l1: int = Field(default=16, ge=16, le=10000)
    quote_age_ms: int = Field(default=1000, ge=1, le=60000)
    stale_seconds: int = Field(default=30, ge=1, le=300)
    raw_path: Path = Path("data/first4-order-flow")
    max_storage_bytes: int = Field(default=2_000_000_000, ge=1_000_000)
    min_free_bytes: int = Field(default=1_000_000_000, ge=1_000_000)
    queue_events: int = Field(default=8192, ge=16, le=65536)
    batch_events: int = Field(default=512, ge=1, le=2048)
    flush_seconds: float = Field(default=0.25, ge=0.05, le=1)
    max_segments_per_stock: int = Field(default=16, ge=1, le=32)

    @model_validator(mode="after")
    def bounded_batch(self) -> "OrderFlowConfig":
        if self.batch_events > self.queue_events:
            raise ValueError("order_flow batch_events must not exceed queue_events")
        return self

    def stock_capacity(self) -> int:
        cost = 2 if self.feed_mode == "TBT_TRADES_TBT_QUOTES" else 1
        capacity = max(0, self.available_tbt - self.reserved_tbt) // cost
        if self.feed_mode == "TBT_TRADES_L1_QUOTES":
            capacity = min(capacity, max(0, self.available_l1 - self.reserved_l1))
        return min(self.max_stocks, capacity)


class First4Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    method: Literal["FIRST4_PRIOR15_Q5_98P_102C_20260923"] = "FIRST4_PRIOR15_Q5_98P_102C_20260923"
    environment: Literal["PAPER"] = "PAPER"
    expected_account: Literal["DUP655399"] = "DUP655399"
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=4003, ge=1, le=65535)
    client_id: Literal[81] = 81
    order_flow: OrderFlowConfig = Field(default_factory=OrderFlowConfig)
    armed: bool = False
    # Explicit, single-session permission to arm only after the opening check.
    arm_after_quote_check_on: date | None = None
    # No production defaults are inferred from synthetic economics.
    expiry_rule: Literal["NEAREST_WITHIN_24H_LATER_TIE"] | None = None
    strike_rule: Literal["NEAREST_STRICT_OTM_WITHIN_1PCT"] | None = None
    premium_budget_usd: Literal[250] | None = None
    fee_reserve_per_package_usd: Literal[10] | None = None
    session_allocation_usd: Literal[1040] = 1040
    packages_per_candidate: Literal[1] = 1
    entry_limit: Literal["SUM_OF_ASKS"] | None = None
    quote_max_age_seconds: Literal[5] | None = None
    entry_deadline_seconds: Literal[180] | None = None
    exit_seconds_before_close: Literal[120] | None = None
    exit_order: Literal["MARKET"] | None = None

    def missing(self) -> list[str]:
        return [
            name
            for name in (
                "expiry_rule",
                "strike_rule",
                "premium_budget_usd",
                "fee_reserve_per_package_usd",
                "entry_limit",
                "quote_max_age_seconds",
                "entry_deadline_seconds",
                "exit_seconds_before_close",
                "exit_order",
            )
            if getattr(self, name) is None
        ]

    def require_settings(self) -> None:
        if self.missing():
            raise ValueError("Missing execution settings: " + ", ".join(self.missing()))

    def require_execution(self) -> None:
        self.require_settings()
        if not self.armed:
            raise ValueError("PAPER entries are unarmed")

    def number(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise ValueError("Missing execution setting: " + name)
        return float(value)


def load(path: Path) -> First4Config:
    config = First4Config.model_validate(yaml.safe_load(path.read_text()))
    if config.arm_after_quote_check_on:
        config.require_settings()
        if config.armed:
            raise ValueError("Opening verification requires armed: false")
    if config.armed:
        config.require_execution()
    return config
