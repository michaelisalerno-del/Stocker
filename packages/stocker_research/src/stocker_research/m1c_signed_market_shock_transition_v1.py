"""Predictor-only calibration helpers for M1C Signed Market Shock Transition V1."""

from __future__ import annotations

import math
from typing import Final, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from stocker_prospective.signed_market_shock_v1 import (
    FROZEN_SHOCK_CHECKPOINTS_V1,
    CheckpointShockThresholdsV1,
    assert_unprotected_sessions_v1,
)

M1C_HIGH_MOVEMENT_THRESHOLD_V1: Final[float] = 0.488333710794033
MINIMUM_PREDICTOR_SUPPORT_V1: Final[int] = 20
ResponseQuintileV1 = Literal["Q1", "Q2", "Q3", "Q4", "Q5", "UNKNOWN_INCOMPLETE"]


class FrozenResponseQuintilesV1(BaseModel):
    """Outcome-free quintile boundaries for the fixed continuous response score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    q20_v1: float | None
    q40_v1: float | None
    q60_v1: float | None
    q80_v1: float | None
    support_v1: int
    calibration_complete_v1: bool
    calibration_missing_reason_v1: str | None

    @model_validator(mode="after")
    def _valid_boundaries(self) -> FrozenResponseQuintilesV1:
        boundaries = (self.q20_v1, self.q40_v1, self.q60_v1, self.q80_v1)
        if self.support_v1 < 0:
            raise ValueError("response-quintile support cannot be negative")
        if self.calibration_complete_v1:
            if any(value is None or not math.isfinite(value) for value in boundaries):
                raise ValueError("complete response quintiles require finite boundaries")
            observed = tuple(float(value) for value in boundaries if value is not None)
            if observed != tuple(sorted(observed)):
                raise ValueError("response-quintile boundaries must be ordered")
            if self.calibration_missing_reason_v1 is not None:
                raise ValueError("complete response quintiles cannot have a missing reason")
        elif self.calibration_missing_reason_v1 is None:
            raise ValueError("incomplete response quintiles require a missing reason")
        return self


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    return cast(pd.Series, numeric.loc[np.isfinite(numeric.to_numpy(float))])


def _quantile_or_none(values: pd.Series, quantile: float) -> float | None:
    if len(values) < MINIMUM_PREDICTOR_SUPPORT_V1:
        return None
    return float(np.quantile(values.to_numpy(float), quantile, method="linear"))


def freeze_checkpoint_thresholds_v1(
    market_windows: pd.DataFrame,
) -> dict[int, CheckpointShockThresholdsV1]:
    """Freeze only the fixed 2024 market-predictor percentiles by checkpoint."""

    required = {
        "session",
        "checkpoint",
        "complete_v1",
        "market_return_w0_v1",
        "market_range_w0_v1",
        "market_return_w1_v1",
        "market_range_w1_v1",
    }
    missing = sorted(required.difference(market_windows.columns))
    if missing:
        raise ValueError(f"market-window calibration columns missing: {missing}")
    assert_unprotected_sessions_v1(market_windows["session"])
    sessions = market_windows["session"].astype(str)
    development = market_windows.loc[
        sessions.between("2024-01-01", "2024-12-31"),
        list(required),
    ].copy()
    result: dict[int, CheckpointShockThresholdsV1] = {}
    predictor_columns = (
        "market_return_w0_v1",
        "market_range_w0_v1",
        "market_return_w1_v1",
        "market_range_w1_v1",
    )
    for checkpoint in FROZEN_SHOCK_CHECKPOINTS_V1:
        group = development.loc[
            pd.to_numeric(development["checkpoint"], errors="coerce").eq(checkpoint)
        ]
        complete_rows = group["complete_v1"].eq(True)  # noqa: E712
        finite = np.isfinite(
            group.loc[:, predictor_columns].apply(
                pd.to_numeric,
                errors="coerce",
            ).to_numpy(float)
        ).all(axis=1)
        group = group.loc[complete_rows & finite]
        values = {column: _finite_series(group, column) for column in predictor_columns}
        supports = {column: int(len(series)) for column, series in values.items()}
        insufficient = [
            f"{column}={supports[column]}"
            for column in predictor_columns
            if supports[column] < MINIMUM_PREDICTOR_SUPPORT_V1
        ]
        calibration_complete = not insufficient
        result[checkpoint] = CheckpointShockThresholdsV1(
            checkpoint=checkpoint,
            market_return_w0_q10_v1=_quantile_or_none(
                values["market_return_w0_v1"],
                0.10,
            ),
            market_return_w0_q90_v1=_quantile_or_none(
                values["market_return_w0_v1"],
                0.90,
            ),
            market_range_w0_q75_v1=_quantile_or_none(
                values["market_range_w0_v1"],
                0.75,
            ),
            market_return_w1_q10_v1=_quantile_or_none(
                values["market_return_w1_v1"],
                0.10,
            ),
            market_return_w1_q90_v1=_quantile_or_none(
                values["market_return_w1_v1"],
                0.90,
            ),
            market_range_w1_q75_v1=_quantile_or_none(
                values["market_range_w1_v1"],
                0.75,
            ),
            market_return_w0_support_v1=supports["market_return_w0_v1"],
            market_range_w0_support_v1=supports["market_range_w0_v1"],
            market_return_w1_support_v1=supports["market_return_w1_v1"],
            market_range_w1_support_v1=supports["market_range_w1_v1"],
            calibration_complete_v1=calibration_complete,
            calibration_missing_reason_v1=(
                None
                if calibration_complete
                else "insufficient_predictor_support:" + ",".join(insufficient)
            ),
        )
    return result


def freeze_response_quintiles_v1(
    predictor_rows: pd.DataFrame,
) -> FrozenResponseQuintilesV1:
    """Freeze score quintiles from valid 2024 high-M1C onset predictors only."""

    required = {
        "session",
        "M1C_probability",
        "market_shock_state_v1",
        "shock_response_complete_v1",
        "shock_relative_response_v1",
    }
    missing = sorted(required.difference(predictor_rows.columns))
    if missing:
        raise ValueError(f"response-quintile calibration columns missing: {missing}")
    assert_unprotected_sessions_v1(predictor_rows["session"])
    sessions = predictor_rows["session"].astype(str)
    probabilities = pd.to_numeric(predictor_rows["M1C_probability"], errors="coerce")
    valid = (
        sessions.between("2024-01-01", "2024-12-31")
        & probabilities.ge(M1C_HIGH_MOVEMENT_THRESHOLD_V1)
        & predictor_rows["market_shock_state_v1"].isin(
            ["NEGATIVE_SHOCK_ONSET", "POSITIVE_SHOCK_ONSET"]
        )
        & predictor_rows["shock_response_complete_v1"].astype(bool)
    )
    values = _finite_series(
        predictor_rows.loc[valid, ["shock_relative_response_v1"]],
        "shock_relative_response_v1",
    )
    support = int(len(values))
    if support == 0:
        return FrozenResponseQuintilesV1(
            q20_v1=None,
            q40_v1=None,
            q60_v1=None,
            q80_v1=None,
            support_v1=support,
            calibration_complete_v1=False,
            calibration_missing_reason_v1=(
                "no_valid_predictor_rows"
            ),
        )
    observed = values.to_numpy(float)
    boundaries = np.quantile(
        observed,
        [0.20, 0.40, 0.60, 0.80],
        method="linear",
    )
    return FrozenResponseQuintilesV1(
        q20_v1=float(boundaries[0]),
        q40_v1=float(boundaries[1]),
        q60_v1=float(boundaries[2]),
        q80_v1=float(boundaries[3]),
        support_v1=support,
        calibration_complete_v1=True,
        calibration_missing_reason_v1=None,
    )


def assign_response_quintile_v1(
    value: float | None,
    frozen: FrozenResponseQuintilesV1,
) -> ResponseQuintileV1:
    """Assign one score with inclusive upper boundaries and no refitting."""

    if (
        value is None
        or not math.isfinite(float(value))
        or not frozen.calibration_complete_v1
    ):
        return "UNKNOWN_INCOMPLETE"
    boundaries = (frozen.q20_v1, frozen.q40_v1, frozen.q60_v1, frozen.q80_v1)
    if any(boundary is None for boundary in boundaries):
        return "UNKNOWN_INCOMPLETE"
    observed = float(value)
    q20, q40, q60, q80 = (float(boundary) for boundary in boundaries if boundary is not None)
    if observed <= q20:
        return "Q1"
    if observed <= q40:
        return "Q2"
    if observed <= q60:
        return "Q3"
    if observed <= q80:
        return "Q4"
    return "Q5"


__all__ = [
    "FrozenResponseQuintilesV1",
    "M1C_HIGH_MOVEMENT_THRESHOLD_V1",
    "MINIMUM_PREDICTOR_SUPPORT_V1",
    "ResponseQuintileV1",
    "assign_response_quintile_v1",
    "freeze_checkpoint_thresholds_v1",
    "freeze_response_quintiles_v1",
]
