"""Written research hypothesis definitions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

REGISTERED_TEMPLATES = {
    "moving_average_momentum",
    "pullback_in_uptrend",
    "mean_reversion_after_large_down_day",
    "volatility_breakout",
}


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


class HypothesisWalkForward(BaseModel):
    """Stage 3 walk-forward settings."""

    mode: Literal["rolling", "expanding", "fixed"] = "rolling"
    train_bars: int = Field(gt=0)
    test_bars: int = Field(gt=0)
    embargo_bars: int = Field(default=0, ge=0)
    step_bars: int | None = Field(default=None, gt=0)
    minimum_rows: int = Field(default=0, ge=0)

    def to_validation_method(self) -> ValidationMethod:
        """Return the legacy walk-forward config shape."""

        return ValidationMethod(
            mode=self.mode,
            train_size=self.train_bars,
            test_size=self.test_bars,
            step_size=self.step_bars,
            embargo_bars=self.embargo_bars,
            min_rows=self.minimum_rows,
        )


class MinimumEvidence(BaseModel):
    """Minimum evidence thresholds before an idea can be interesting."""

    min_trades: int = Field(default=20, ge=0)
    min_profitable_split_pct: float = Field(default=0.6, ge=0.0, le=1.0)
    min_stability_score: float = Field(default=0.5, ge=0.0, le=1.0)


class HypothesisHoldingPolicy(BaseModel):
    """Preferred holding style and stricter swing evidence gates."""

    preferred_style: Literal["intraday", "swing"] = "intraday"
    allow_intraday: bool = True
    allow_overnight: Literal[False, "conditional", True] = "conditional"
    allow_weekend: Literal[False, "exceptional_only", True] = "exceptional_only"
    max_holding_sessions: int = Field(default=5, ge=1)
    flatten_before_close_minutes: int = Field(default=10, ge=0)
    entry_cutoff_before_close_minutes: int = Field(default=30, ge=0)
    require_exceptional_evidence_for_swing: bool = True
    require_gap_risk_report: bool = True
    min_swing_excess_vs_benchmark: float = 0.05
    min_swing_excess_vs_null: float = 0.03
    min_swing_trade_count: int = Field(default=50, ge=0)
    max_gap_return_contribution_pct: float = Field(default=0.25, ge=0.0)
    max_weekend_exposure_count: int = Field(default=0, ge=0)
    max_overnight_exposure_count: int = Field(default=0, ge=0)
    max_swing_drawdown: float = Field(default=0.15, gt=0.0)


class Hypothesis(BaseModel):
    """A reproducible written trading hypothesis."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    hypothesis_version: int = Field(default=1, ge=1)
    market_universe: str = Field(default="unspecified", min_length=1)
    instrument_type: str = Field(min_length=1)
    symbol_filter: str = Field(default="*", min_length=1)
    timeframe: str = Field(min_length=1)
    data_source: str = Field(default="manual", min_length=1)
    template: str = Field(default="", min_length=1)
    signal_family: Literal[
        "moving_average_momentum",
        "pullback_in_uptrend",
        "mean_reversion_after_large_down_day",
        "volatility_breakout",
    ] = "moving_average_momentum"
    entry_logic: str = Field(min_length=1)
    exit_logic: str = Field(min_length=1)
    holding_period: str = Field(min_length=1)
    direction: Literal["long_only", "short_only", "long_short"] = "long_only"
    costs: HypothesisCostModel
    cost_model: HypothesisCostModel = Field(default_factory=HypothesisCostModel)
    risk_model: HypothesisRiskModel = Field(default_factory=HypothesisRiskModel)
    risk: dict[str, Any] = Field(default_factory=dict)
    parameter_space: dict[str, list[int | float | str | bool]]
    maximum_parameter_sets: int = Field(default=100, ge=1, le=1000)
    validation_method: ValidationMethod
    walkforward: HypothesisWalkForward
    expected_edge_reason: str = Field(min_length=1)
    invalidation_rules: list[str] = Field(min_length=1)
    minimum_evidence: MinimumEvidence = Field(default_factory=MinimumEvidence)
    holding_policy: HypothesisHoldingPolicy = Field(default_factory=HypothesisHoldingPolicy)
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _compatibility_defaults(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        value = dict(raw)
        if not value.get("template") and value.get("signal_family"):
            value["template"] = value["signal_family"]
        if not value.get("signal_family") and value.get("template"):
            value["signal_family"] = value["template"]
        if "costs" not in value and "cost_model" in value:
            value["costs"] = value["cost_model"]
        if value.get("costs") is None:
            raise ValueError("Hypothesis must include a cost model")
        if "cost_model" not in value and "costs" in value:
            value["cost_model"] = value["costs"]
        if "risk_model" not in value and isinstance(value.get("risk"), dict):
            risk = value["risk"]
            value["risk_model"] = {
                "max_drawdown": risk.get("max_drawdown", 0.25),
                "min_trades": risk.get("min_trades", 20),
            }
        if "walkforward" not in value and "validation_method" in value:
            method = value["validation_method"]
            if isinstance(method, dict):
                value["walkforward"] = {
                    "mode": method.get("mode", "rolling"),
                    "train_bars": method.get("train_size"),
                    "test_bars": method.get("test_size"),
                    "step_bars": method.get("step_size"),
                    "embargo_bars": method.get("embargo_bars", 0),
                    "minimum_rows": method.get("min_rows", 0),
                }
        if "validation_method" not in value and "walkforward" in value:
            wf = value["walkforward"]
            if isinstance(wf, dict):
                value["validation_method"] = {
                    "type": "walk_forward",
                    "mode": wf.get("mode", "rolling"),
                    "train_size": wf.get("train_bars"),
                    "test_size": wf.get("test_bars"),
                    "step_size": wf.get("step_bars"),
                    "embargo_bars": wf.get("embargo_bars", 0),
                    "min_rows": wf.get("minimum_rows", 0),
                }
        return value

    @field_validator("template")
    @classmethod
    def _template_registered(cls, value: str) -> str:
        if value not in REGISTERED_TEMPLATES:
            raise ValueError(f"template must be registered: {value}")
        return value

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

    @model_validator(mode="after")
    def _stage3_contract(self) -> Hypothesis:
        total = 1
        for choices in self.parameter_space.values():
            total *= len(choices)
        if total > self.maximum_parameter_sets:
            raise ValueError(
                f"parameter_space expands to {total}, exceeding maximum_parameter_sets "
                f"{self.maximum_parameter_sets}"
            )
        if not self.expected_edge_reason.strip():
            raise ValueError("expected_edge_reason must be non-empty")
        if not self.invalidation_rules:
            raise ValueError("invalidation_rules must be non-empty")
        return self


def load_hypothesis(path: str | Path) -> Hypothesis:
    """Load a hypothesis YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Hypothesis file must contain a YAML mapping: {path}")
    return Hypothesis.model_validate(raw)
