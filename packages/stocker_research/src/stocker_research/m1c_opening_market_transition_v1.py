"""Predictor-only helpers for M1C Opening Market Transition V1."""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from stocker_prospective.opening_market_transition_v1 import (
    OPENING_TRANSITION_CHECKPOINT_V1,
    OpeningTransitionThresholdsV1,
)
from stocker_prospective.signed_market_shock_v1 import (
    assert_unprotected_sessions_v1,
)

M1C_HIGH_MOVEMENT_THRESHOLD_V1: Final[float] = 0.488333710794033
MINIMUM_PREDICTOR_SUPPORT_V1: Final[int] = 20
OpeningResponseQuintileV1 = Literal[
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "UNKNOWN_INCOMPLETE",
]
PopulationKeyV1 = tuple[str, str, str, int]


class FrozenOpeningResponseQuintilesV1(BaseModel):
    """Outcome-free quintile boundaries for relative opening response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    q20_v1: float | None
    q40_v1: float | None
    q60_v1: float | None
    q80_v1: float | None
    support_v1: int
    calibration_complete_v1: bool
    calibration_missing_reason_v1: str | None

    @model_validator(mode="after")
    def _valid_boundaries(
        self,
    ) -> FrozenOpeningResponseQuintilesV1:
        boundaries = (self.q20_v1, self.q40_v1, self.q60_v1, self.q80_v1)
        if self.support_v1 < 0:
            raise ValueError("opening response-quintile support cannot be negative")
        if self.calibration_complete_v1:
            if any(value is None or not math.isfinite(value) for value in boundaries):
                raise ValueError(
                    "complete opening response quintiles require finite boundaries"
                )
            observed = tuple(float(value) for value in boundaries if value is not None)
            if observed != tuple(sorted(observed)):
                raise ValueError(
                    "opening response-quintile boundaries must be ordered"
                )
            if self.calibration_missing_reason_v1 is not None:
                raise ValueError(
                    "complete opening response quintiles cannot have a missing reason"
                )
        elif self.calibration_missing_reason_v1 is None:
            raise ValueError(
                "incomplete opening response quintiles require a missing reason"
            )
        return self


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    return cast(pd.Series, numeric.loc[np.isfinite(numeric.to_numpy(float))])


def _quantile_or_none(values: pd.Series, quantile: float) -> float | None:
    if len(values) < MINIMUM_PREDICTOR_SUPPORT_V1:
        return None
    return float(np.quantile(values.to_numpy(float), quantile, method="linear"))


def freeze_opening_thresholds_v1(
    market_predictors: pd.DataFrame,
) -> OpeningTransitionThresholdsV1:
    """Freeze the fixed checkpoint-6 percentiles from complete 2024 predictors."""

    predictor_columns = (
        "market_opening_return_v1",
        "market_opening_range_v1",
        "market_overnight_gap_v1",
        "market_total_transition_v1",
    )
    required = {"session", "checkpoint", "complete_v1", *predictor_columns}
    missing = sorted(required.difference(market_predictors.columns))
    if missing:
        raise ValueError(f"opening calibration columns missing: {missing}")
    assert_unprotected_sessions_v1(market_predictors["session"])
    sessions = market_predictors["session"].astype(str)
    checkpoints = pd.to_numeric(market_predictors["checkpoint"], errors="coerce")
    development = market_predictors.loc[
        sessions.between("2024-01-01", "2024-12-31")
        & checkpoints.eq(OPENING_TRANSITION_CHECKPOINT_V1)
        & market_predictors["complete_v1"].eq(True),  # noqa: E712
        list(predictor_columns),
    ].copy()
    finite = np.isfinite(
        development.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ).all(axis=1)
    development = development.loc[finite]
    values = {
        column: _finite_series(development, column)
        for column in predictor_columns
    }
    supports = {column: int(len(series)) for column, series in values.items()}
    insufficient = [
        f"{column}={supports[column]}"
        for column in predictor_columns
        if supports[column] < MINIMUM_PREDICTOR_SUPPORT_V1
    ]
    calibration_complete = not insufficient
    return OpeningTransitionThresholdsV1(
        market_opening_return_q10_v1=_quantile_or_none(
            values["market_opening_return_v1"],
            0.10,
        ),
        market_opening_return_q90_v1=_quantile_or_none(
            values["market_opening_return_v1"],
            0.90,
        ),
        market_opening_range_q75_v1=_quantile_or_none(
            values["market_opening_range_v1"],
            0.75,
        ),
        market_overnight_gap_q10_v1=_quantile_or_none(
            values["market_overnight_gap_v1"],
            0.10,
        ),
        market_overnight_gap_q90_v1=_quantile_or_none(
            values["market_overnight_gap_v1"],
            0.90,
        ),
        market_total_transition_q10_v1=_quantile_or_none(
            values["market_total_transition_v1"],
            0.10,
        ),
        market_total_transition_q90_v1=_quantile_or_none(
            values["market_total_transition_v1"],
            0.90,
        ),
        market_opening_return_support_v1=supports[
            "market_opening_return_v1"
        ],
        market_opening_range_support_v1=supports["market_opening_range_v1"],
        market_overnight_gap_support_v1=supports["market_overnight_gap_v1"],
        market_total_transition_support_v1=supports[
            "market_total_transition_v1"
        ],
        calibration_complete_v1=calibration_complete,
        calibration_missing_reason_v1=(
            None
            if calibration_complete
            else "insufficient_predictor_support:" + ",".join(insufficient)
        ),
    )


def freeze_opening_response_quintiles_v1(
    predictor_rows: pd.DataFrame,
) -> FrozenOpeningResponseQuintilesV1:
    """Freeze score quintiles from valid 2024 high-M1C severe predictors."""

    required = {
        "session",
        "checkpoint",
        "tail_phase_v1",
        "M1C_probability",
        "opening_market_transition_state_v1",
        "stock_opening_response_complete_v1",
        "stock_relative_opening_response_v1",
    }
    missing = sorted(required.difference(predictor_rows.columns))
    if missing:
        raise ValueError(
            f"opening response-quintile calibration columns missing: {missing}"
        )
    assert_unprotected_sessions_v1(predictor_rows["session"])
    sessions = predictor_rows["session"].astype(str)
    checkpoints = pd.to_numeric(predictor_rows["checkpoint"], errors="coerce")
    probabilities = pd.to_numeric(
        predictor_rows["M1C_probability"],
        errors="coerce",
    )
    valid = (
        sessions.between("2024-01-01", "2024-12-31")
        & checkpoints.eq(OPENING_TRANSITION_CHECKPOINT_V1)
        & predictor_rows["tail_phase_v1"].eq("FIRST_ENTRY")
        & probabilities.ge(M1C_HIGH_MOVEMENT_THRESHOLD_V1)
        & predictor_rows["opening_market_transition_state_v1"].isin(
            [
                "NEGATIVE_SEVERE_OPENING_TRANSITION",
                "POSITIVE_SEVERE_OPENING_TRANSITION",
            ]
        )
        & predictor_rows["stock_opening_response_complete_v1"].astype(bool)
    )
    values = _finite_series(
        predictor_rows.loc[valid, ["stock_relative_opening_response_v1"]],
        "stock_relative_opening_response_v1",
    )
    support = int(len(values))
    if support == 0:
        return FrozenOpeningResponseQuintilesV1(
            q20_v1=None,
            q40_v1=None,
            q60_v1=None,
            q80_v1=None,
            support_v1=0,
            calibration_complete_v1=False,
            calibration_missing_reason_v1="no_valid_predictor_rows",
        )
    boundaries = np.quantile(
        values.to_numpy(float),
        [0.20, 0.40, 0.60, 0.80],
        method="linear",
    )
    return FrozenOpeningResponseQuintilesV1(
        q20_v1=float(boundaries[0]),
        q40_v1=float(boundaries[1]),
        q60_v1=float(boundaries[2]),
        q80_v1=float(boundaries[3]),
        support_v1=support,
        calibration_complete_v1=True,
        calibration_missing_reason_v1=None,
    )


def assign_opening_response_quintile_v1(
    value: float | None,
    frozen: FrozenOpeningResponseQuintilesV1,
) -> OpeningResponseQuintileV1:
    """Assign one score using frozen inclusive upper boundaries."""

    if (
        value is None
        or not math.isfinite(float(value))
        or not frozen.calibration_complete_v1
    ):
        return "UNKNOWN_INCOMPLETE"
    boundaries = (frozen.q20_v1, frozen.q40_v1, frozen.q60_v1, frozen.q80_v1)
    if any(boundary is None for boundary in boundaries):
        return "UNKNOWN_INCOMPLETE"
    q20, q40, q60, q80 = (
        float(boundary) for boundary in boundaries if boundary is not None
    )
    observed = float(value)
    if observed <= q20:
        return "Q1"
    if observed <= q40:
        return "Q2"
    if observed <= q60:
        return "Q3"
    if observed <= q80:
        return "Q4"
    return "Q5"


def validate_prior_population_reconciliation_v1(
    reconciliation: pd.DataFrame,
    *,
    expected_tail_keys: Collection[PopulationKeyV1],
    expected_primary_episode_ids: Collection[str],
) -> None:
    """Require an explicit one-row explanation for every prior Tail candidate."""

    required = {
        "period",
        "stock",
        "session",
        "checkpoint",
        "fresh_episode_id",
        "tail_phase_v1",
        "market_shock_state_v1",
        "included_in_primary_signed_shock_population_v1",
        "included_in_tail_phase_diagnostics_v1",
        "inclusion_exclusion_reason_v1",
    }
    missing = sorted(required.difference(reconciliation.columns))
    if missing:
        raise ValueError(f"population reconciliation columns missing: {missing}")
    keys = [
        (
            str(row.period),
            str(row.stock),
            str(row.session)[:10],
            int(cast(Any, row).checkpoint),
        )
        for row in reconciliation.itertuples(index=False)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("population reconciliation has duplicate tail diagnostic keys")
    if set(keys) != set(expected_tail_keys):
        raise ValueError("population reconciliation tail diagnostic keys differ")
    if not reconciliation[
        "included_in_tail_phase_diagnostics_v1"
    ].astype(bool).all():
        raise ValueError("population reconciliation contains a non-Tail row")
    reasons = reconciliation["inclusion_exclusion_reason_v1"].fillna("").astype(str)
    if reasons.str.strip().eq("").any():
        raise ValueError("population reconciliation requires an exact reason")
    included = reconciliation.loc[
        reconciliation[
            "included_in_primary_signed_shock_population_v1"
        ].astype(bool),
        "fresh_episode_id",
    ]
    included_ids = set(included.dropna().astype(str))
    if included.isna().any() or included_ids != set(expected_primary_episode_ids):
        raise ValueError("population reconciliation primary episode IDs differ")


__all__ = [
    "FrozenOpeningResponseQuintilesV1",
    "M1C_HIGH_MOVEMENT_THRESHOLD_V1",
    "MINIMUM_PREDICTOR_SUPPORT_V1",
    "OpeningResponseQuintileV1",
    "assign_opening_response_quintile_v1",
    "freeze_opening_response_quintiles_v1",
    "freeze_opening_thresholds_v1",
    "validate_prior_population_reconciliation_v1",
]
