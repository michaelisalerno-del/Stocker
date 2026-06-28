"""Written research hypothesis definitions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class HypothesisCostModel(BaseModel):
    """Cost assumptions in basis points."""

    spread_bps: float = Field(default=0.0, ge=0.0)
    commission_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)

    def one_way_bps(self) -> float:
        """Return one-way cost in basis points."""

        return self.spread_bps + self.commission_bps + self.slippage_bps

    def round_trip_bps(self) -> float:
        """Return round-trip cost in basis points."""

        return self.one_way_bps() * 2


class HypothesisRiskModel(BaseModel):
    """Simple research-side rejection thresholds."""

    max_drawdown: float = Field(default=0.25, gt=0.0)
    min_trades: int = Field(default=20, ge=0)


class ValidationMethod(BaseModel):
    """Walk-forward validation settings."""

    type: Literal["walk_forward"] = "walk_forward"
    mode: Literal["rolling", "expanding", "fixed"] = "rolling"
    train_size: int = Field(gt=0)
    test_size: int = Field(gt=0)
    step_size: int | None = Field(default=None, gt=0)
    embargo_bars: int = Field(default=0, ge=0)
    min_rows: int = Field(default=0, ge=0)


class Hypothesis(BaseModel):
    """A reproducible written trading hypothesis."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    market_universe: str = Field(min_length=1)
    instrument_type: str = Field(min_length=1)
    symbol_filter: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    signal_family: Literal[
        "moving_average_momentum",
        "mean_reversion_after_large_down_day",
        "volatility_breakout",
    ]
    entry_logic: str = Field(min_length=1)
    exit_logic: str = Field(min_length=1)
    holding_period: str = Field(min_length=1)
    direction: Literal["long_only", "short_only", "long_short"] = "long_only"
    cost_model: HypothesisCostModel = Field(default_factory=HypothesisCostModel)
    risk_model: HypothesisRiskModel = Field(default_factory=HypothesisRiskModel)
    parameter_space: dict[str, list[int | float | str | bool]]
    validation_method: ValidationMethod
    expected_edge_reason: str = Field(min_length=1)
    invalidation_rules: list[str] = Field(min_length=1)
    created_at: datetime

    @field_validator("parameter_space")
    @classmethod
    def _parameter_space_not_empty(
        cls, value: dict[str, list[int | float | str | bool]]
    ) -> dict[str, list[int | float | str | bool]]:
        if not value:
            raise ValueError("parameter_space must not be empty")
        empty = [name for name, choices in value.items() if not choices]
        if empty:
            raise ValueError(f"parameter_space entries must not be empty: {', '.join(empty)}")
        return value


def load_hypothesis(path: str | Path) -> Hypothesis:
    """Load a hypothesis YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Hypothesis file must contain a YAML mapping: {path}")
    return Hypothesis.model_validate(raw)
