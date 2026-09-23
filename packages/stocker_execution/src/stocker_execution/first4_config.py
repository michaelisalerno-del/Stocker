"""One PAPER account and the explicit research-to-listed-contract choices."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PAPER_ACCOUNT = "DUP655399"


class First4Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    method: Literal["FIRST4_PRIOR15_Q5_98P_102C_20260923"] = "FIRST4_PRIOR15_Q5_98P_102C_20260923"
    environment: Literal["PAPER"] = "PAPER"
    expected_account: Literal["DUP655399"] = "DUP655399"
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=4003, ge=1, le=65535)
    client_id: Literal[81] = 81
    armed: bool = False
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

    def require_execution(self) -> None:
        if self.missing():
            raise ValueError("Missing execution settings: " + ", ".join(self.missing()))
        if not self.armed:
            raise ValueError("PAPER entries are unarmed")

    def number(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise ValueError("Missing execution setting: " + name)
        return float(value)


def load(path: Path) -> First4Config:
    config = First4Config.model_validate(yaml.safe_load(path.read_text()))
    if config.armed:
        config.require_execution()
    return config
