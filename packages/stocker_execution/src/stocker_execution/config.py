"""One account, one frozen method, and explicit listed-product authorisations."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

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
    available_market_data_lines: int | None = Field(default=None, ge=16, le=75)
    market_data_allocation_source: str | None = Field(default=None, min_length=10)


def load(path: Path) -> FuturesConfig:
    return FuturesConfig.model_validate(yaml.safe_load(path.read_text()))
