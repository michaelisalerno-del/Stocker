"""Causal pair-to-family economic lifecycle aggregation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .taxonomy import FamilyTaxonomy

_PAIR_FIELDS = {
    "period",
    "score_session",
    "decision_timestamp",
    "prediction_frozen_at",
    "feature_max_availability_timestamp",
    "training_latest_availability_timestamp",
    "loop_id",
    "orientation",
    "horizon",
    "edge_state",
    "p_edge_positive",
    "p_edge_active",
    "p_change_now",
    "p_on_next",
    "p_off_next",
    "p_survive_horizon",
    "posterior_mean_net_bps",
    "posterior_lower_bound_net_bps",
    "posterior_std_net_bps",
    "posterior_run_length_mean",
    "effective_sessions",
    "independent_stocks",
    "effective_sample_size",
    "reason_codes",
}


def _weighted(values: pd.Series, weights: np.ndarray) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = np.isfinite(numeric) & np.isfinite(weights)
    if not finite.any():
        return math.nan
    return float(np.average(numeric[finite], weights=weights[finite]))


def _family_state(states: set[str]) -> str:
    if "active" in states:
        return "active"
    if "decaying" in states:
        return "decaying"
    if "retired" in states and "unknown" not in states:
        return "retired"
    return "unknown"


def aggregate_family_states(
    pair_states: pd.DataFrame,
    taxonomy: FamilyTaxonomy,
) -> pd.DataFrame:
    """Aggregate frozen pair states without importing hindsight or future columns."""

    missing = sorted(_PAIR_FIELDS - set(pair_states.columns))
    if missing:
        raise ValueError(f"missing pair-state columns: {missing}")
    source = taxonomy.map_pairs(pair_states.loc[:, sorted(_PAIR_FIELDS)])
    source = source.loc[source["family_mapping_status"].eq("mapped")].copy()
    for column in (
        "decision_timestamp",
        "prediction_frozen_at",
        "feature_max_availability_timestamp",
        "training_latest_availability_timestamp",
    ):
        source[column] = pd.to_datetime(source[column], utc=True, errors="coerce")
    if not source["decision_timestamp"].eq(source["prediction_frozen_at"]).all():
        raise ValueError("pair forecast freeze differs from its decision timestamp")
    future_feature = source["feature_max_availability_timestamp"].notna() & source[
        "feature_max_availability_timestamp"
    ].gt(source["prediction_frozen_at"])
    future_training = source["training_latest_availability_timestamp"].notna() & source[
        "training_latest_availability_timestamp"
    ].ge(source["prediction_frozen_at"])
    if future_feature.any() or future_training.any():
        raise ValueError("future pair evidence enters family aggregation")

    keys = ["period", "score_session", "destination_family", "horizon"]
    rows: list[dict[str, object]] = []
    for key, group in source.groupby(keys, sort=True, observed=True):
        freezes = group["prediction_frozen_at"].dropna().unique()
        if len(freezes) != 1:
            raise ValueError("mapped pair freezes disagree inside family/session")
        weights = np.clip(
            pd.to_numeric(group["effective_sample_size"], errors="coerce")
            .fillna(1.0)
            .to_numpy(float),
            1.0,
            50.0,
        )
        states = set(group["edge_state"].astype(str))
        row: dict[str, object] = {
            "period": int(str(key[0])),
            "score_session": str(key[1]),
            "destination_family": str(key[2]),
            "horizon": int(str(key[3])),
            "forecast_freeze_timestamp": pd.Timestamp(freezes[0]),
            "feature_availability_timestamp": group["feature_max_availability_timestamp"].max(),
            "training_cutoff": group["training_latest_availability_timestamp"].max(),
            "operational_state": _family_state(states),
            "mapped_pair_count": int(len(group)),
            "mapped_pair_ids": "|".join(
                sorted(
                    f"{loop_id}|{orientation}"
                    for loop_id, orientation in group[["loop_id", "orientation"]].itertuples(
                        index=False, name=None
                    )
                )
            ),
            "max_p_edge_active": float(group["p_edge_active"].max()),
            "mean_p_edge_active": _weighted(group["p_edge_active"], weights),
            "max_p_edge_positive": float(group["p_edge_positive"].max()),
            "mean_p_edge_positive": _weighted(group["p_edge_positive"], weights),
            "max_p_on_next": float(group["p_on_next"].max()),
            "mean_p_on_next": _weighted(group["p_on_next"], weights),
            "mean_p_off_next": _weighted(group["p_off_next"], weights),
            "mean_p_change_now": _weighted(group["p_change_now"], weights),
            "mean_p_survive_horizon": _weighted(group["p_survive_horizon"], weights),
            "posterior_mean_net_bps": _weighted(group["posterior_mean_net_bps"], weights),
            "posterior_lower_bound_net_bps": _weighted(
                group["posterior_lower_bound_net_bps"], weights
            ),
            "posterior_std_net_bps": _weighted(group["posterior_std_net_bps"], weights),
            "posterior_run_length_mean": _weighted(group["posterior_run_length_mean"], weights),
            "effective_sessions": float(group["effective_sessions"].max()),
            "independent_stocks": int(group["independent_stocks"].max()),
            "effective_sample_size": float(group["effective_sample_size"].sum()),
            "reason_codes": "|".join(
                sorted(
                    {
                        code
                        for values in group["reason_codes"].fillna("").astype(str)
                        for code in values.split("|")
                        if code
                    }
                )
            ),
            "hindsight_labels_used_as_features": False,
        }
        rows.append(row)
    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["period", "destination_family", "score_session"], kind="stable"
    ).reset_index(drop=True)
    group_keys = ["period", "destination_family"]
    result["active_probability_change"] = result.groupby(group_keys, sort=False, observed=True)[
        "max_p_edge_active"
    ].diff()
    result["posterior_mean_change_bps"] = result.groupby(group_keys, sort=False, observed=True)[
        "posterior_mean_net_bps"
    ].diff()
    ages: list[int] = []
    since_active: list[int] = []
    for _, group in result.groupby(group_keys, sort=False, observed=True):
        age = 0
        prior: str | None = None
        inactive_age = 0
        for state in group["operational_state"].astype(str):
            age = age + 1 if state == prior else 1
            prior = state
            if state == "active":
                inactive_age = 0
            else:
                inactive_age += 1
            ages.append(age)
            since_active.append(inactive_age)
    result["state_age_sessions"] = ages
    result["sessions_since_active"] = since_active
    return result


def derive_source_events(family_states: pd.DataFrame) -> pd.DataFrame:
    """Timestamp family lifecycle changes using the current frozen observation only."""

    required = {
        "period",
        "score_session",
        "destination_family",
        "operational_state",
        "forecast_freeze_timestamp",
    }
    missing = sorted(required - set(family_states.columns))
    if missing:
        raise ValueError(f"missing family-state columns: {missing}")
    result = family_states.copy().sort_values(
        ["period", "destination_family", "score_session"], kind="stable"
    )
    result["forecast_freeze_timestamp"] = pd.to_datetime(
        result["forecast_freeze_timestamp"], utc=True, errors="raise"
    )
    keys = ["period", "destination_family"]
    result["previous_operational_state"] = result.groupby(keys, sort=False, observed=True)[
        "operational_state"
    ].shift(1)
    current = result["operational_state"].astype(str)
    previous = result["previous_operational_state"].fillna("unknown").astype(str)
    result["source_active"] = current.eq("active")
    result["newly_active"] = current.eq("active") & previous.ne("active")
    result["newly_decaying"] = current.eq("decaying") & previous.eq("active")
    result["newly_retired"] = current.eq("retired") & previous.isin(["active", "decaying"])
    result["source_state_transition"] = current.ne(previous)
    result["source_event_timestamp"] = result["forecast_freeze_timestamp"]
    return result.reset_index(drop=True)


__all__ = ["aggregate_family_states", "derive_source_events"]
