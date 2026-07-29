"""Typed research-only virtual position ledger projections."""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

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
    option_schedule_status: (
        Literal[
            "scheduled",
            "streaming",
            "complete",
            "rejected",
            "expired",
        ]
        | None
    )
    con_id: int | None
    expiry: date | None
    dte: Literal[1] | None
    strike: float | None
    quantity: Literal[1]
    multiplier: int | None = Field(default=None, gt=0)
    planned_live_market_data_lines: Literal[2] | None
    pair_outcome_count: int = Field(ge=0, le=2)
    pair_complete_count: int = Field(ge=0, le=2)
    predicted_outcome_present: bool
    opposite_outcome_present: bool
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
    exit_convention: Literal["first_valid_live_bid_at_or_after_frozen_15m_horizon"]
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


class QuietStateVirtualLegV1(BaseModel):
    """Exact immutable quote evidence for one quiet short-premium leg."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    side: Literal["short", "long"]
    con_id: int = Field(gt=0)
    expiry: date
    dte: int = Field(ge=0)
    dte_bucket: Literal["0DTE", "1DTE", "3_TO_5_DTE"]
    strike: float = Field(gt=0.0)
    right: Literal["C", "P"]
    multiplier: int = Field(gt=0)
    target_delta: float | None
    entry_quote_timestamp_utc: datetime | None
    entry_bid: float | None
    entry_ask: float | None
    entry_fill_price: float | None
    exit_quote_timestamp_utc: datetime | None
    exit_bid: float | None
    exit_ask: float | None
    exit_fill_price: float | None

    @model_validator(mode="after")
    def _fill_prices_match_the_conservative_side(self) -> QuietStateVirtualLegV1:
        expected_entry = self.entry_bid if self.side == "short" else self.entry_ask
        expected_exit = self.exit_ask if self.side == "short" else self.exit_bid
        for label, observed, expected in (
            ("entry", self.entry_fill_price, expected_entry),
            ("exit", self.exit_fill_price, expected_exit),
        ):
            if observed is None and expected is None:
                continue
            if (
                observed is None
                or expected is None
                or not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(f"quiet {label} fill differs from conservative quote side")
        return self


class QuietStateCaptureContractV1(BaseModel):
    """Latest diagnostic quote state for one bounded quiet option contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    option_contract_id: int = Field(gt=0)
    con_id: int | None = Field(default=None, gt=0)
    expiry: date
    dte: int = Field(ge=0)
    dte_bucket: Literal["0DTE", "1DTE", "3_TO_5_DTE"]
    strike: float = Field(gt=0.0)
    right: Literal["C", "P"]
    multiplier: int = Field(gt=0)
    selection_roles: tuple[str, ...]
    resolution_status: str
    rejection_reason: str | None
    recording_started_at_utc: datetime | None
    recording_ends_at_utc: datetime | None
    latest_quote_received_at_utc: datetime | None
    latest_bid: float | None
    latest_ask: float | None
    latest_market_data_type: str | None
    latest_recording_status: str | None
    latest_quote_quality_flags: tuple[str, ...]


class QuietStateVirtualCaptureV1(BaseModel):
    """Truthful scheduled/capturing state before immutable quiet outcomes exist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    virtual_capture_id: str = Field(min_length=1)
    ledger_scope: Literal["quiet_state_short_premium_capture"]
    run_id: str
    observation_id: str
    observation_kind: Literal["quiet_bottom_10"]
    session_date: date
    symbol: str
    trigger_timestamp_utc: datetime
    entry_timestamp_utc: datetime
    lifecycle_state: VirtualPositionLifecycle
    status_reason: str | None
    option_plan_recorded: bool
    requested_contract_count: int = Field(ge=0)
    selected_contract_count: int = Field(ge=0)
    option_plan_capacity_reduced: bool
    option_plan_missing_buckets: tuple[str, ...]
    completion_status: str
    completed_at_utc: datetime | None
    frozen_short_premium_outcome_count: int = Field(ge=0)
    contracts: tuple[QuietStateCaptureContractV1, ...]
    scientific_recording_valid: bool
    latest_quotes_are_diagnostic_only: Literal[True]
    execution_claimed: Literal[False]
    paper_fill_claimed: Literal[False]

    @model_validator(mode="after")
    def _capture_state_is_consistent(self) -> QuietStateVirtualCaptureV1:
        if self.lifecycle_state is VirtualPositionLifecycle.SCHEDULED and (
            self.option_plan_recorded or self.contracts
        ):
            raise ValueError("scheduled quiet capture already has a contract plan")
        if self.lifecycle_state is VirtualPositionLifecycle.CAPTURING and (
            not self.option_plan_recorded or not self.contracts
        ):
            raise ValueError("capturing quiet state lacks its bounded contract plan")
        if self.lifecycle_state is VirtualPositionLifecycle.CLOSED and (
            self.completion_status != "complete" or self.frozen_short_premium_outcome_count == 0
        ):
            raise ValueError("closed quiet capture lacks frozen structure outcomes")
        if self.lifecycle_state is VirtualPositionLifecycle.INVALID and not self.status_reason:
            raise ValueError("invalid quiet capture lacks a persisted reason")
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
    legs: tuple[QuietStateVirtualLegV1, ...]
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
        if self.lifecycle_state == VirtualPositionLifecycle.CLOSED:
            self._validate_closed_structure()
        return self

    def _validate_closed_structure(self) -> None:
        if any(
            value is None
            for leg in self.legs
            for value in (
                leg.entry_quote_timestamp_utc,
                leg.entry_fill_price,
                leg.exit_quote_timestamp_utc,
                leg.exit_fill_price,
            )
        ):
            raise ValueError("closed quiet virtual position has incomplete leg quotes")
        if len({leg.con_id for leg in self.legs}) != len(self.legs):
            raise ValueError("quiet virtual position repeats a contract")
        if (
            len({leg.expiry for leg in self.legs}) != 1
            or len({leg.dte for leg in self.legs}) != 1
            or {leg.dte_bucket for leg in self.legs} != {self.dte_bucket}
            or len({leg.multiplier for leg in self.legs}) != 1
        ):
            raise ValueError("quiet virtual position legs do not share one expiry contract")
        short_calls = [leg for leg in self.legs if leg.side == "short" and leg.right == "C"]
        short_puts = [leg for leg in self.legs if leg.side == "short" and leg.right == "P"]
        long_calls = [leg for leg in self.legs if leg.side == "long" and leg.right == "C"]
        long_puts = [leg for leg in self.legs if leg.side == "long" and leg.right == "P"]
        if self.structure_type in {"ATM_IRON_BUTTERFLY", "DELTA_IRON_CONDOR"}:
            if not all(
                len(group) == 1 for group in (short_calls, short_puts, long_calls, long_puts)
            ):
                raise ValueError("quiet four-leg structure composition is invalid")
            if not (
                long_puts[0].strike < short_puts[0].strike
                and short_calls[0].strike < long_calls[0].strike
            ):
                raise ValueError("quiet four-leg protective strike geometry is invalid")
            if self.structure_type == "ATM_IRON_BUTTERFLY" and (
                not math.isclose(
                    short_calls[0].strike,
                    short_puts[0].strike,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    long_calls[0].strike - short_calls[0].strike,
                    short_puts[0].strike - long_puts[0].strike,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("quiet iron butterfly is not symmetric around its shorts")
        elif self.structure_type == "CALL_CREDIT_SPREAD":
            if (
                len(short_calls) != 1
                or len(long_calls) != 1
                or len(self.legs) != 2
                or long_calls[0].strike <= short_calls[0].strike
            ):
                raise ValueError("quiet call credit spread composition is invalid")
        elif (
            len(short_puts) != 1
            or len(long_puts) != 1
            or len(self.legs) != 2
            or long_puts[0].strike >= short_puts[0].strike
        ):
            raise ValueError("quiet put credit spread composition is invalid")

        multiplier = self.legs[0].multiplier
        opening_credit = (
            sum(
                (1.0 if leg.side == "short" else -1.0) * float(leg.entry_fill_price or 0.0)
                for leg in self.legs
            )
            * multiplier
        )
        closing_debit = (
            sum(
                (1.0 if leg.side == "short" else -1.0) * float(leg.exit_fill_price or 0.0)
                for leg in self.legs
            )
            * multiplier
        )
        if (
            self.opening_net_credit is None
            or self.closing_net_debit is None
            or self.conservative_pnl is None
            or not math.isclose(
                self.opening_net_credit,
                opening_credit,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.closing_net_debit,
                closing_debit,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.conservative_pnl,
                opening_credit - closing_debit,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("quiet net P&L differs from immutable leg quote evidence")


__all__ = [
    "OpeningReversalVirtualPositionV1",
    "QuietStateCaptureContractV1",
    "QuietStateVirtualLegV1",
    "QuietStateVirtualCaptureV1",
    "QuietStateVirtualPositionV1",
    "VirtualPositionLifecycle",
]
