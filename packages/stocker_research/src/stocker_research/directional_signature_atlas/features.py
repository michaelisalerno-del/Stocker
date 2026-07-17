"""Causal feature checks, state motifs, and contemporaneous ranks."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pandas as pd

_FORBIDDEN_EXACT = {
    "month",
    "symbol",
    "symbol_norm",
    "stock_identity",
    "future_state",
    "future_route",
    "target",
    "payoff",
    "outcome",
    "outcome_label",
    "first_touch_target",
    "net_long_return_bps",
    "net_short_return_bps",
    "gross_long_return_bps",
    "gross_short_return_bps",
    "realised_child",
    "realized_child",
    "realised_morph",
    "realized_morph",
}
_FORBIDDEN_FRAGMENTS = (
    "future_return",
    "future_direction",
    "future_state",
    "future_loop",
    "route_identity",
    "child_identity",
    "morph_identity",
    "realised_morph",
    "realized_morph",
    "mfe",
    "mae",
    "hindsight_episode",
    "outcome_episode",
    "terminal_payoff",
    "actual_target",
    "outcome_label",
    "payoff",
    "target",
    "outcome",
)


def assert_outcome_free_feature_names(feature_names: Iterable[str]) -> None:
    """Reject identity, outcome, and future-route fields from a causal ledger."""

    invalid = []
    for feature in feature_names:
        lowered = feature.lower()
        if lowered in _FORBIDDEN_EXACT or any(token in lowered for token in _FORBIDDEN_FRAGMENTS):
            invalid.append(feature)
    if invalid:
        raise ValueError(f"forbidden causal feature(s): {sorted(invalid)}")


def assert_causal_feature_ledger(
    frame: pd.DataFrame,
    feature_names: Iterable[str],
    *,
    decision_column: str = "decision_timestamp",
) -> None:
    """Verify every feature has an explicit non-future availability timestamp."""

    names = list(feature_names)
    assert_outcome_free_feature_names(names)
    decisions = pd.to_datetime(frame[decision_column], utc=True, errors="raise")
    missing = [name for name in names if f"{name}__available_at" not in frame]
    if missing:
        raise AssertionError(f"missing feature availability columns: {missing}")
    for name in names:
        availability = pd.to_datetime(frame[f"{name}__available_at"], utc=True, errors="coerce")
        populated = frame[name].notna()
        if availability.loc[populated].isna().any():
            raise AssertionError(f"missing availability for populated feature {name}")
        if availability.loc[populated].gt(decisions.loc[populated]).any():
            raise AssertionError(f"future availability detected for feature {name}")


def _repeat_count(states: list[int]) -> int:
    if len(states) < 3:
        return 0
    count = 0
    cursor = len(states) - 1
    while cursor >= 2 and states[cursor] == states[cursor - 2]:
        count += 1
        cursor -= 2
    return count


def reconstruct_state_motifs(
    frame: pd.DataFrame,
    *,
    symbol_column: str = "symbol",
    session_column: str = "session",
    timestamp_column: str = "timestamp",
    state_column: str = "state",
) -> pd.DataFrame:
    """Attach causal run-state motifs of lengths two, three, and four.

    Motifs contain completed runs only.  The active state is exported
    separately and does not enter history until a later causal transition
    completes that run.
    """

    required = {symbol_column, session_column, timestamp_column, state_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"state motif input missing columns: {missing}")
    output = frame.copy()
    output["state_motif_2"] = pd.NA
    output["state_motif_3"] = pd.NA
    output["state_motif_4"] = pd.NA
    output["previous_state"] = pd.NA
    output["previous_state_2"] = pd.NA
    output["state_age_bars"] = pd.NA
    output["prior_completed_state_dwell_bars"] = pd.NA
    output["prior_completed_transition_duration_bars"] = pd.NA
    output["same_orientation_repeat_count"] = 0
    grouped = output.sort_values(timestamp_column, kind="mergesort").groupby(
        [symbol_column, session_column], sort=False
    )
    for _, positions in grouped.groups.items():
        run_states: list[int] = []
        run_lengths: list[int] = []
        for position in list(positions):
            state = int(cast(Any, output.at[position, state_column]))
            if not run_states or state != run_states[-1]:
                run_states.append(state)
                run_lengths.append(1)
            else:
                run_lengths[-1] += 1
            output.at[position, "state_age_bars"] = run_lengths[-1]
            if len(run_states) >= 2:
                output.at[position, "previous_state"] = run_states[-2]
                output.at[position, "prior_completed_state_dwell_bars"] = run_lengths[-2]
                output.at[position, "prior_completed_transition_duration_bars"] = pd.NA
            if len(run_states) >= 3:
                output.at[position, "previous_state_2"] = run_states[-3]
            completed_states = run_states[:-1]
            for length in (2, 3, 4):
                if len(completed_states) >= length:
                    output.at[position, f"state_motif_{length}"] = ">".join(
                        str(value) for value in completed_states[-length:]
                    )
            output.at[position, "same_orientation_repeat_count"] = _repeat_count(run_states)
    for column in (
        "state_age_bars",
        "prior_completed_state_dwell_bars",
        "prior_completed_transition_duration_bars",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("Int64")
    return output.sort_index()


def fit_training_quantile_bins(
    frame: pd.DataFrame,
    feature: str,
    *,
    discovery_periods: set[int],
    bins: int,
    period_column: str = "period",
) -> list[float]:
    """Fit coarse quantile edges using discovery rows only."""

    if bins < 2 or bins > 5:
        raise ValueError("training-only quantile bins must be between two and five")
    periods = pd.to_numeric(frame[period_column], errors="raise").astype(int)
    values = pd.to_numeric(frame.loc[periods.isin(discovery_periods), feature], errors="coerce")
    values = values.dropna()
    if values.empty:
        raise ValueError(f"no discovery values for {feature}")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    return [float(value) for value in np.unique(values.quantile(quantiles).to_numpy(float))]


def add_cross_sectional_features(
    frame: pd.DataFrame,
    return_columns: Iterable[str],
    *,
    timestamp_column: str = "timestamp",
    min_peers: int = 10,
) -> pd.DataFrame:
    """Create ranks and equal-stock context from contemporaneous non-missing peers."""

    if min_peers < 2:
        raise ValueError("cross-sectional ranks need at least two peers")
    output = frame.copy()

    def breadth(values: pd.Series) -> float:
        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.count() < min_peers:
            return math.nan
        return float(numeric_values.gt(0).mean())

    for column in return_columns:
        numeric = pd.to_numeric(output[column], errors="coerce")
        groups = numeric.groupby(output[timestamp_column], sort=False)
        count = groups.transform("count")
        rank = groups.rank(method="average")
        denominator = (count - 1).replace(0, np.nan)
        ranked = (rank - 1) / denominator
        output[f"{column}_cross_sectional_rank"] = ranked.where(count.ge(min_peers))
        mean = groups.transform("mean").where(count.ge(min_peers))
        output[f"{column}_universe_mean"] = mean
        output[f"{column}_versus_universe"] = (numeric - mean).where(numeric.notna())
        output[f"{column}_breadth_positive"] = groups.transform(breadth)
        output[f"{column}_dispersion"] = groups.transform("std").where(count.ge(min_peers))
    return output
