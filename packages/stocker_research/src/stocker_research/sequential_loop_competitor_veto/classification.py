"""Training-only economic payoff-family classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PayoffClassConfig:
    """Frozen V2-compatible support and uncertainty requirements."""

    model_name: str = "hierarchical_payoff_history_change_point"
    upper_bound_z: float = 1.6448536269514722
    minimum_independent_sessions: float = 8.0
    minimum_independent_stocks: int = 5
    minimum_effective_sample_size: float = 12.0
    maximum_posterior_std_bps: float = 80.0


def _forbidden_outcome_columns(columns: pd.Index) -> list[str]:
    exact = {
        "net_payoff_bps",
        "robust_net_payoff_bps",
        "target_payoff_positive",
        "payoff_positive",
        "realised_loop",
        "realized_loop",
        "loop_occurs",
        "hindsight_episode_state",
    }
    return sorted(
        str(column)
        for column in columns
        if str(column) in exact or str(column).startswith("target_")
    )


def classify_payoff_families(
    forecasts: pd.DataFrame,
    config: PayoffClassConfig,
    *,
    score_session: str | None = None,
) -> pd.DataFrame:
    """Classify loop/orientation cells from immutable pre-session forecasts only."""

    forbidden = _forbidden_outcome_columns(forecasts.columns)
    if forbidden:
        raise ValueError(f"outcome column is forbidden in classification: {forbidden}")
    frame = forecasts.loc[forecasts["model_name"].eq(config.model_name)].copy()
    if score_session is not None:
        frame = frame.loc[frame["score_session"].astype(str).eq(str(score_session))].copy()
    if frame.empty:
        return frame.assign(
            posterior_upper_bound_net_bps=pd.Series(dtype=float),
            payoff_class_support_pass=pd.Series(dtype=bool),
            payoff_class=pd.Series(dtype=str),
        )
    keys = ["period", "score_session", "loop_id", "orientation", "horizon"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate immutable payoff-family forecast")
    decision = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="raise")
    frozen = pd.to_datetime(frame["prediction_frozen_at"], utc=True, errors="raise")
    if frozen.gt(decision).any():
        raise ValueError("forecast freeze occurs after scoring decision")
    training = pd.to_datetime(
        frame["training_latest_availability_timestamp"], utc=True, errors="coerce"
    )
    if training.notna().any() and training[training.notna()].ge(decision[training.notna()]).any():
        raise ValueError("training availability must strictly precede scoring decision")

    frame["posterior_upper_bound_net_bps"] = pd.to_numeric(
        frame["posterior_mean_net_bps"], errors="raise"
    ) + config.upper_bound_z * pd.to_numeric(frame["posterior_std_net_bps"], errors="raise")
    support = (
        pd.to_numeric(frame["effective_sessions"], errors="coerce").ge(
            config.minimum_independent_sessions
        )
        & pd.to_numeric(frame["independent_stocks"], errors="coerce").ge(
            config.minimum_independent_stocks
        )
        & pd.to_numeric(frame["effective_sample_size"], errors="coerce").ge(
            config.minimum_effective_sample_size
        )
        & pd.to_numeric(frame["posterior_std_net_bps"], errors="coerce").le(
            config.maximum_posterior_std_bps
        )
    )
    mean = pd.to_numeric(frame["posterior_mean_net_bps"], errors="coerce")
    lower = pd.to_numeric(frame["posterior_lower_bound_net_bps"], errors="coerce")
    upper = pd.to_numeric(frame["posterior_upper_bound_net_bps"], errors="coerce")
    frame["payoff_class_support_pass"] = support
    frame["payoff_class"] = np.select(
        [support & mean.gt(0.0) & lower.gt(0.0), support & upper.lt(0.0)],
        ["good", "bad"],
        default="unknown",
    )
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)
