"""Frozen causal feature surface for the primary ranker."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stocker_research.observable_event_ranking_v1.contract import PRIMARY_FEATURES
from stocker_research.observable_event_ranking_v1.events import NEW_YORK, robust_location_scale
from stocker_research.observable_event_ranking_v1.provenance import (
    assert_primary_feature_columns,
)

_FEATURE_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "name": "event_strength",
        "definition": "minimum of causal market and sector relative acceleration z-scores",
    },
    {
        "name": "market_relative_return_5m",
        "definition": "stock five-minute close return minus leave-one-out eligible-market median",
    },
    {
        "name": "market_relative_return_15m",
        "definition": "stock recent fifteen-minute return minus leave-one-out market median",
    },
    {
        "name": "market_relative_return_30m",
        "definition": "stock recent thirty-minute return minus leave-one-out market median",
    },
    {
        "name": "sector_relative_return_15m",
        "definition": "stock recent fifteen-minute return minus leave-one-out sector median",
    },
    {
        "name": "sector_relative_return_30m",
        "definition": "stock recent thirty-minute return minus leave-one-out sector median",
    },
    {
        "name": "market_relative_acceleration_z",
        "definition": "market-relative acceleration robust-scaled on prior sixty sessions only",
    },
    {
        "name": "sector_relative_acceleration_z",
        "definition": "sector-relative acceleration robust-scaled on prior sixty sessions only",
    },
    {
        "name": "realized_volatility_30m",
        "definition": "population standard deviation of six causal five-minute log returns",
    },
    {
        "name": "activity_shock_z",
        "definition": (
            "close times provider volume activity proxy robust-scaled using prior sixty-session "
            "stock-by-clock history only"
        ),
    },
    {
        "name": "distance_from_session_high",
        "definition": "current close divided by causal session high minus one",
    },
    {
        "name": "session_fraction",
        "definition": "elapsed regular-session seconds divided by scheduled session seconds",
    },
)


def feature_manifest() -> dict[str, Any]:
    """Return the frozen machine-readable feature manifest."""

    names = tuple(item["name"] for item in _FEATURE_DEFINITIONS)
    if names != PRIMARY_FEATURES:
        raise RuntimeError("feature definition order differs from frozen contract")
    return {
        "manifest_version": "observable_event_ranking_v1_features",
        "feature_count": len(_FEATURE_DEFINITIONS),
        "features": [dict(item) for item in _FEATURE_DEFINITIONS],
        "imputation": "training_only_median",
        "clipping": "training_only_0.5_and_99.5_percentiles",
        "standardisation": "training_only_mean_and_population_standard_deviation",
        "provider_volume_interpretation": "provider_reported_activity_proxy",
    }


def build_feature_ledger(events: pd.DataFrame) -> pd.DataFrame:
    """Project event rows onto frozen identifiers, metadata, and primary features."""

    missing = sorted(set(PRIMARY_FEATURES).difference(events.columns))
    if missing:
        raise ValueError(f"event rows missing frozen features: {missing}")
    assert_primary_feature_columns(PRIMARY_FEATURES)
    metadata = [
        column
        for column in (
            "event_id",
            "slate_id",
            "symbol",
            "sector",
            "session",
            "assigned_decision_time",
        )
        if column in events
    ]
    ledger = events.loc[:, [*metadata, *PRIMARY_FEATURES]].copy()
    return ledger.sort_values(
        [column for column in ("assigned_decision_time", "slate_id", "symbol") if column in ledger],
        kind="mergesort",
    ).reset_index(drop=True)


def primary_matrix(feature_ledger: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the identifier-free primary design matrix in frozen column order."""

    missing = sorted(set(PRIMARY_FEATURES).difference(feature_ledger.columns))
    if missing:
        raise ValueError(f"feature ledger missing columns: {missing}")
    assert_primary_feature_columns(PRIMARY_FEATURES)
    return feature_ledger.loc[:, PRIMARY_FEATURES].to_numpy(dtype="float64"), PRIMARY_FEATURES


def add_causal_price_activity_features(
    bars: pd.DataFrame,
    *,
    trailing_sessions: int = 60,
    min_activity_observations: int = 20,
) -> pd.DataFrame:
    """Add causal price/activity features with prior stock-by-clock activity scaling."""

    required = {
        "symbol",
        "session",
        "session_open",
        "bar_start",
        "bar_end",
        "feature_availability_time",
        "session_close",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "fully_completed",
        "gap_status",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"causal feature bars missing columns: {missing}")
    frame = bars.copy()
    for column in (
        "session",
        "session_open",
        "bar_start",
        "bar_end",
        "feature_availability_time",
        "session_close",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame = frame.sort_values(["symbol", "session", "bar_end"], kind="mergesort").reset_index(
        drop=True
    )
    frame["provider_dollar_activity_proxy"] = frame["close"].astype("float64") * frame[
        "volume"
    ].astype("float64")
    frame["bar_clock"] = frame["bar_end"].dt.tz_convert(NEW_YORK).dt.strftime("%H:%M")
    numeric = frame[["open", "high", "low", "close", "volume"]].astype("float64")
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=1),
        index=frame.index,
    )
    possible_ohlc = (
        numeric["open"].gt(0.0)
        & numeric["high"].ge(numeric[["open", "close"]].max(axis=1))
        & numeric["low"].le(numeric[["open", "close"]].min(axis=1))
        & numeric["low"].gt(0.0)
        & numeric["high"].ge(numeric["low"])
        & numeric["volume"].ge(0.0)
    )
    frame["bar_context_valid"] = (
        finite
        & possible_ohlc
        & frame["fully_completed"].eq(True)
        & frame["gap_status"].eq("complete")
        & frame["bar_end"].sub(frame["bar_start"]).eq(pd.Timedelta(minutes=5))
        & frame["feature_availability_time"].ge(frame["bar_end"])
        & frame["bar_start"].ge(frame["session_open"])
        & frame["bar_end"].le(frame["session_close"])
    )
    frame["feature_context_valid"] = False
    frame["feature_context_reason"] = "unvalidated"
    frame["realized_volatility_30m"] = np.nan
    frame["distance_from_session_high"] = np.nan
    frame["session_fraction"] = np.nan
    for (_, _), group in frame.groupby(["symbol", "session"], sort=True, observed=True):
        declared_opens = group["session_open"].drop_duplicates()
        declared_closes = group["session_close"].drop_duplicates()
        schedule_is_unique = len(declared_opens) == 1 and len(declared_closes) == 1
        session_open = group["session_open"].iloc[0]
        session_close = group["session_close"].iloc[0]
        begins_at_open = bool(group["bar_start"].iloc[0] == session_open)
        contiguous = group["bar_start"].eq(group["bar_end"].shift(1))
        contiguous.iloc[0] = begins_at_open
        row_valid = frame.loc[group.index, "bar_context_valid"] & contiguous
        causal_chain_valid = row_valid.cummin().astype(bool)
        frame.loc[group.index, "feature_context_valid"] = causal_chain_valid.to_numpy()
        reasons = np.where(
            causal_chain_valid,
            "valid",
            np.where(
                frame.loc[group.index, "bar_context_valid"].to_numpy(dtype=bool),
                "missing_open_or_noncontiguous_history",
                "invalid_incomplete_or_unavailable_bar",
            ),
        )
        if not schedule_is_unique:
            causal_chain_valid[:] = False
            reasons[:] = "inconsistent_session_schedule"
            frame.loc[group.index, "feature_context_valid"] = False
        frame.loc[group.index, "feature_context_reason"] = reasons
        log_prices = pd.Series(
            np.log(group["close"].astype("float64").to_numpy()),
            index=group.index,
        )
        log_returns = log_prices.diff()
        realized = log_returns.rolling(6, min_periods=6).std(ddof=0)
        realized.loc[~causal_chain_valid] = np.nan
        frame.loc[group.index, "realized_volatility_30m"] = realized.to_numpy()
        session_high = group["high"].astype("float64").cummax()
        distance = group["close"].astype("float64") / session_high - 1.0
        distance.loc[~causal_chain_valid] = np.nan
        frame.loc[group.index, "distance_from_session_high"] = distance.to_numpy()
        total_seconds = (session_close - session_open).total_seconds()
        elapsed = (group["bar_end"] - session_open).dt.total_seconds()
        fraction = elapsed / total_seconds if total_seconds > 0.0 else np.nan
        fraction = pd.Series(np.clip(fraction, 0.0, 1.0), index=group.index)
        fraction.loc[~causal_chain_valid] = np.nan
        frame.loc[group.index, "session_fraction"] = fraction.to_numpy()
    frame["activity_shock_z"] = np.nan
    for _symbol, symbol_rows in frame.groupby("symbol", sort=True, observed=True):
        sessions = list(pd.Index(symbol_rows["session"].drop_duplicates()).sort_values())
        for session_position, session in enumerate(sessions):
            prior_sessions = sessions[
                max(0, session_position - trailing_sessions) : session_position
            ]
            prior = symbol_rows.loc[
                symbol_rows["session"].isin(prior_sessions) & symbol_rows["feature_context_valid"]
            ]
            current = symbol_rows.loc[
                symbol_rows["session"].eq(session) & symbol_rows["feature_context_valid"]
            ]
            for index, row in current.iterrows():
                same_clock = prior.loc[prior["bar_clock"].eq(row["bar_clock"])]
                if len(same_clock) < min_activity_observations:
                    continue
                location, scale = robust_location_scale(
                    same_clock["provider_dollar_activity_proxy"].to_numpy(dtype="float64")
                )
                frame.at[index, "activity_shock_z"] = (
                    float(row["provider_dollar_activity_proxy"]) - location
                ) / scale
    return frame
