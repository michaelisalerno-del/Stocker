"""Secondary contemporaneous cross-sectional relative outcomes and baseline."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd


def construct_relative_outcomes(
    absolute: pd.DataFrame,
    *,
    minimum_peers: int = 10,
) -> pd.DataFrame:
    """Rank future equal-universe residual returns at identical decision times."""

    output = absolute.copy()
    gross = pd.to_numeric(output["gross_long_return_bps"], errors="coerce")
    groups = gross.groupby(output["decision_timestamp"], sort=False)
    peer_count = groups.transform("count")
    universe = groups.transform("mean").where(peer_count.ge(minimum_peers))
    residual = (gross - universe).where(gross.notna())
    rank = residual.groupby(output["decision_timestamp"], sort=False).rank(method="average")
    percentile = (rank - 1.0) / (peer_count - 1.0).replace(0.0, np.nan)
    target = np.where(
        percentile.ge(0.80),
        "LONG",
        np.where(percentile.le(0.20), "SHORT", "NEUTRAL"),
    )
    target = np.where(peer_count.ge(minimum_peers) & residual.notna(), target, "UNAVAILABLE")
    return pd.DataFrame(
        {
            "opportunity_id": output["opportunity_id"],
            "period": output["period"],
            "session": output["session"],
            "symbol": output["symbol"],
            "decision_clock": output["decision_clock"],
            "decision_timestamp": output["decision_timestamp"],
            "peer_count": peer_count,
            "future_equal_universe_return_bps": universe,
            "future_residual_return_bps": residual,
            "future_residual_percentile": percentile,
            "target": target,
            "long_net_bps": residual,
            "short_net_bps": -residual,
            "round_trip_cost_bps": 0.0,
            "sector_relative_status": "unavailable_no_frozen_sector_membership",
            "absolute_profitability_claim_allowed": False,
        }
    )


def relative_strength_baseline(relative: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    joined = relative.merge(
        features[["opportunity_id", "return_6_bps_cross_sectional_rank"]],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
    )
    rank = pd.to_numeric(joined["return_6_bps_cross_sectional_rank"], errors="coerce")
    state = np.where(rank.ge(0.80), "LONG", np.where(rank.le(0.20), "SHORT", "NEUTRAL"))
    state = np.where(rank.notna(), state, "NEUTRAL")
    return joined[["opportunity_id", "period", "session", "symbol", "decision_clock"]].assign(
        model_id="contemporaneous_relative_strength", predicted_state=state
    )


def relative_baseline_economic_metrics(
    predictions: pd.DataFrame,
    relative: pd.DataFrame,
) -> pd.DataFrame:
    joined = predictions.merge(
        relative[["opportunity_id", "target", "long_net_bps", "short_net_bps"]],
        on="opportunity_id",
        how="inner",
        validate="one_to_one",
    )
    joined = joined.loc[joined["target"].ne("UNAVAILABLE")]
    rows: list[dict[str, Any]] = []
    for period, group in joined.groupby("period", sort=True):
        state = group["predicted_state"].astype(str)
        directional = state.isin(["LONG", "SHORT"])
        payoff = np.where(
            state.eq("LONG"),
            group["long_net_bps"],
            np.where(state.eq("SHORT"), group["short_net_bps"], 0.0),
        ).astype(float)
        negative = abs(float(payoff[payoff < 0.0].sum()))
        rows.append(
            {
                "model_id": "contemporaneous_relative_strength",
                "period": int(cast(Any, period)),
                "rows": len(group),
                "coverage": float(directional.mean()),
                "mean_residual_bps_per_directional_output": float(payoff[directional].mean())
                if directional.any()
                else math.nan,
                "mean_residual_bps_per_opportunity": float(payoff.mean()),
                "total_residual_bps": float(payoff.sum()),
                "hit_rate": float((payoff[directional] > 0.0).mean())
                if directional.any()
                else math.nan,
                "profit_factor": float(payoff[payoff > 0.0].sum()) / negative
                if negative > 0.0
                else math.nan,
                "absolute_profitability_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows)
