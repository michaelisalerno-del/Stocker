"""Explicit scientific and operational validity layers for frozen M1C."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class CheckpointValidityLayers(BaseModel):
    """Keep causal computation separate from post-selection feed readiness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    m1c_computation_valid: bool
    m1c_computation_reasons: tuple[str, ...]
    source_transfer_valid: bool
    source_transfer_reasons: tuple[str, ...]
    opening_reversal_prediction_eligible: bool
    opening_reversal_prediction_reasons: tuple[str, ...]
    promotion_eligible: bool
    promotion_reasons: tuple[str, ...]
    option_recording_ready: bool
    option_recording_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def _layers_do_not_retroactively_authorize_inputs(
        self,
    ) -> CheckpointValidityLayers:
        pairs = (
            (self.m1c_computation_valid, self.m1c_computation_reasons, "M1C computation"),
            (self.source_transfer_valid, self.source_transfer_reasons, "source transfer"),
            (
                self.opening_reversal_prediction_eligible,
                self.opening_reversal_prediction_reasons,
                "Opening Reversal prediction",
            ),
            (self.promotion_eligible, self.promotion_reasons, "promotion"),
            (self.option_recording_ready, self.option_recording_reasons, "option recording"),
        )
        for valid, reasons, label in pairs:
            if valid and reasons:
                raise ValueError(f"{label} cannot be valid with rejection reasons")
            if not valid and not reasons:
                raise ValueError(f"{label} requires an explicit rejection reason")
        if self.source_transfer_valid and not self.m1c_computation_valid:
            raise ValueError("source transfer cannot exceed M1C computation validity")
        if self.opening_reversal_prediction_eligible and not self.m1c_computation_valid:
            raise ValueError("prediction eligibility cannot exceed M1C computation validity")
        if self.promotion_eligible and not self.opening_reversal_prediction_eligible:
            raise ValueError("promotion eligibility requires an eligible prediction")
        if self.option_recording_ready and not self.promotion_eligible:
            raise ValueError("option recording readiness requires promotion eligibility")
        return self


__all__ = ["CheckpointValidityLayers"]
