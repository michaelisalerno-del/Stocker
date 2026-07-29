"""Typed research-only virtual position ledger projections."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VirtualPositionLifecycle(StrEnum):
    """States supported by immutable virtual-position evidence projections."""

    SCHEDULED = "SCHEDULED"
    CAPTURING = "CAPTURING"
    CLOSED = "CLOSED"
    INVALID = "INVALID"


class OpeningReversalVirtualPositionV1(BaseModel):
    """One V1.1 predicted 1DTE leg projected from its strict two-line evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    virtual_position_id: str = Field(min_length=1)
    ledger_scope: Literal["opening_reversal_v1_1"]
    run_id: str
    experiment_id: Literal["m1c-prospective-opening-reversal-v1"]
    experiment_version: Literal["1.1"]
    activation_receipt_identity: str
    causal_barrier_audit_identity: str
    promotion_identity: str
    prediction_receipt_hash_v1: str
    episode_id: str
    session_date: date
    symbol: str
    entry_timestamp_utc: datetime
    predicted_direction: Literal["CALL", "PUT"]
    right: Literal["C", "P"]
    role: Literal["predicted_leg"]
    lifecycle_state: VirtualPositionLifecycle
    status_reason: str | None
    con_id: int | None
    expiry: date | None
    dte: Literal[1] | None
    strike: float | None
    quantity: Literal[1]
    multiplier: int | None = Field(default=None, gt=0)
    planned_live_market_data_lines: Literal[2] | None
    pair_outcome_count: int = Field(ge=0, le=2)
    pair_complete_count: int = Field(ge=0, le=2)
    entry_bid: float | None
    entry_ask: float | None
    entry_quote_timestamp_utc: datetime | None
    exit_bid: float | None
    exit_ask: float | None
    exit_quote_timestamp_utc: datetime | None
    conservative_return_v1: float | None
    gross_quote_pnl: float | None
    latest_observed_bid: float | None
    latest_observed_ask: float | None
    latest_quote_received_at_utc: datetime | None
    latest_quote_recording_status: str | None
    latest_quote_quality_flags: tuple[str, ...]
    entry_convention: Literal["first_valid_live_ask_at_or_after_entry"]
    exit_convention: Literal["last_valid_live_bid_at_or_before_frozen_15m_horizon"]
    scientific_eligible: Literal[True]
    execution_claimed: Literal[False]
    paper_fill_claimed: Literal[False]

    @model_validator(mode="after")
    def _evidence_is_consistent(self) -> OpeningReversalVirtualPositionV1:
        expected_right = "C" if self.predicted_direction == "CALL" else "P"
        if self.right != expected_right:
            raise ValueError("virtual position right differs from frozen prediction")
        if self.con_id is not None and (
            self.dte != 1
            or self.expiry is None
            or self.strike is None
            or self.planned_live_market_data_lines != 2
        ):
            raise ValueError("virtual position contract is not the strict primary pair")
        if self.lifecycle_state is VirtualPositionLifecycle.CLOSED and (
            self.pair_outcome_count != 2
            or self.pair_complete_count != 2
            or self.entry_ask is None
            or self.entry_ask <= 0.0
            or self.exit_bid is None
            or self.gross_quote_pnl is None
            or self.status_reason is not None
        ):
            raise ValueError("closed virtual position lacks complete paired bid/ask evidence")
        return self


class QuietStateVirtualPositionV1(BaseModel):
    """One frozen quiet-state short-premium structure/horizon outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    virtual_position_id: str = Field(min_length=1)
    ledger_scope: Literal["quiet_state_short_premium"]
    run_id: str
    observation_id: str
    observation_kind: Literal["quiet_bottom_10"]
    session_date: date
    symbol: str
    trigger_timestamp_utc: datetime
    entry_timestamp_utc: datetime
    structure_type: Literal[
        "ATM_IRON_BUTTERFLY",
        "DELTA_IRON_CONDOR",
        "CALL_CREDIT_SPREAD",
        "PUT_CREDIT_SPREAD",
    ]
    dte_bucket: Literal["0DTE", "1DTE", "3_TO_5_DTE"]
    horizon_label: Literal["5m", "10m", "15m", "30m", "60m", "session_end"]
    horizon_minutes: int | None = Field(default=None, ge=0)
    lifecycle_state: Literal[
        VirtualPositionLifecycle.CLOSED,
        VirtualPositionLifecycle.INVALID,
    ]
    status_reason: str | None
    opening_net_credit: float | None
    closing_net_debit: float | None
    conservative_pnl: float | None
    configured_commission_pnl: float | None
    maximum_defined_risk: float | None
    return_on_maximum_risk: float | None
    short_strike_touched: bool | None
    protective_wing_touched: bool | None
    attempted: bool
    complete_quote_quality: bool
    strict_quote_quality: bool
    quality_status: str
    quality_flags: tuple[str, ...]
    legs: tuple[dict[str, Any], ...]
    leg_count: int = Field(ge=0)
    scientific_recording_valid: bool
    scientific_option_evidence: bool
    cohort_phase: str
    conservative_fill_convention: Literal["open_short_bid_long_ask_close_short_ask_long_bid"]
    execution_claimed: Literal[False]
    paper_fill_claimed: Literal[False]

    @model_validator(mode="after")
    def _evidence_is_consistent(self) -> QuietStateVirtualPositionV1:
        if self.leg_count != len(self.legs):
            raise ValueError("quiet virtual position leg count differs from evidence")
        if self.lifecycle_state == VirtualPositionLifecycle.CLOSED and (
            not self.attempted
            or not self.complete_quote_quality
            or self.opening_net_credit is None
            or self.closing_net_debit is None
            or self.conservative_pnl is None
            or self.status_reason is not None
        ):
            raise ValueError("closed quiet virtual position lacks conservative quote evidence")
        return self


__all__ = [
    "OpeningReversalVirtualPositionV1",
    "QuietStateVirtualPositionV1",
    "VirtualPositionLifecycle",
]
