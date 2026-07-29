"""Paired, fixed-rule metrics for clean-anchor price acceptance."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def four_cell_interaction(
    source: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Summarise the pre-registered 2x2 anchor/acceptance cells."""

    frame = source.copy()
    frame["interaction_cell"] = np.select(
        [
            ~frame["static_anchor_veto_pass"] & ~frame["price_acceptance_pass"],
            ~frame["static_anchor_veto_pass"] & frame["price_acceptance_pass"],
            frame["static_anchor_veto_pass"] & ~frame["price_acceptance_pass"],
        ],
        [
            "anchor_fail|acceptance_fail",
            "anchor_fail|acceptance_pass",
            "anchor_pass|acceptance_fail",
        ],
        default="anchor_pass|acceptance_pass",
    )
    keys = [*group_columns, "interaction_cell"]
    records: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        net = pd.to_numeric(group["net_payoff_bps"], errors="coerce").dropna()
        gross = pd.to_numeric(group["gross_payoff_bps"], errors="coerce").dropna()
        costs = pd.to_numeric(group["total_cost_bps"], errors="coerce").dropna()
        record = dict(zip(keys, key_values, strict=True))
        record.update(
            {
                "opportunities": int(len(group)),
                "independent_stocks": int(group["symbol"].nunique()),
                "gross_payoff_bps": float(gross.sum()),
                "net_payoff_bps": float(net.sum()),
                "mean_net_payoff_bps": float(net.mean()),
                "median_net_payoff_bps": float(net.median()),
                "positive_payoff_rate": float(net.gt(0.0).mean()),
                "total_cost_bps": float(costs.sum()),
                "twice_cost_net_payoff_bps": float((gross - 2.0 * costs).sum()),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def veto_accounting(source: pd.DataFrame, *, admitted: pd.Series) -> dict[str, float | int]:
    """Compute losses avoided less profits mistakenly rejected on one population."""

    admitted_mask = admitted.reindex(source.index).fillna(False).astype(bool)
    rejected = pd.to_numeric(source.loc[~admitted_mask, "net_payoff_bps"], errors="coerce")
    retained = pd.to_numeric(source.loc[admitted_mask, "net_payoff_bps"], errors="coerce")
    losses = float(-rejected.loc[rejected.lt(0.0)].sum())
    winners = float(rejected.loc[rejected.gt(0.0)].sum())
    return {
        "opportunities": int(len(source)),
        "admitted_opportunities": int(admitted_mask.sum()),
        "rejected_opportunities": int((~admitted_mask).sum()),
        "losing_opportunities_rejected": int(rejected.lt(0.0).sum()),
        "winning_opportunities_rejected": int(rejected.gt(0.0).sum()),
        "losses_avoided_bps": losses,
        "profits_mistakenly_rejected_bps": winners,
        "veto_value_bps": losses - winners,
        "net_payoff_retained_bps": float(retained.sum()),
        "coverage": float(admitted_mask.mean()) if len(source) else 0.0,
        "abstention": float((~admitted_mask).mean()) if len(source) else 0.0,
        "average_avoided_loss_bps": float(-rejected.loc[rejected.lt(0.0)].mean())
        if rejected.lt(0.0).any()
        else 0.0,
        "average_rejected_winner_bps": float(rejected.loc[rejected.gt(0.0)].mean())
        if rejected.gt(0.0).any()
        else 0.0,
    }


def paired_difference_rows(
    decisions: pd.DataFrame,
    *,
    treatment: str,
    control: str,
) -> pd.DataFrame:
    """Pair policies by immutable opportunity ID and fail on population drift."""

    left = decisions.loc[decisions["variant"].eq(treatment)].copy()
    right = decisions.loc[decisions["variant"].eq(control)].copy()
    left_ids = set(left["opportunity_id"].astype(str))
    right_ids = set(right["opportunity_id"].astype(str))
    if left_ids != right_ids or len(left) != len(left_ids) or len(right) != len(right_ids):
        raise ValueError("paired populations differ")
    keep = [column for column in ["opportunity_id", "period", "session_date"] if column in left]
    paired = left.loc[:, [*keep, "policy_net_payoff_bps"]].merge(
        right.loc[:, ["opportunity_id", "policy_net_payoff_bps"]],
        on="opportunity_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_treatment", "_control"),
    )
    paired["difference_bps"] = pd.to_numeric(
        paired["policy_net_payoff_bps_treatment"], errors="coerce"
    ) - pd.to_numeric(paired["policy_net_payoff_bps_control"], errors="coerce")
    return paired


def paired_variant_comparison(
    decisions: pd.DataFrame,
    *,
    treatment: str,
    control: str,
) -> dict[str, float | int | str]:
    """Return the registered paired economic comparison."""

    paired = paired_difference_rows(decisions, treatment=treatment, control=control)
    difference = paired["difference_bps"].dropna()
    by_session = paired.groupby("session_date", dropna=False)["difference_bps"].sum()
    return {
        "treatment": treatment,
        "control": control,
        "paired_opportunities": int(len(paired)),
        "paired_observable_opportunities": int(len(difference)),
        "paired_total_difference_bps": float(difference.sum()),
        "paired_mean_difference_bps": float(difference.mean()),
        "paired_median_difference_bps": float(difference.median()),
        "sessions_improved_fraction": float(by_session.gt(0.0).mean()),
    }


def session_block_bootstrap(
    differences: pd.DataFrame,
    *,
    resamples: int,
    block_length: int,
    seed: int,
) -> dict[str, float]:
    """Moving-session-block interval for the paired mean policy increment."""

    if resamples <= 0 or block_length <= 0:
        raise ValueError("resamples and block_length must be positive")
    session = (
        differences.groupby(["period", "session_date"], sort=True)["difference_bps"]
        .mean()
        .reset_index()
    )
    observed = float(session["difference_bps"].mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    period_arrays = [
        group["difference_bps"].to_numpy(float) for _, group in session.groupby("period")
    ]
    for draw in range(resamples):
        sampled: list[float] = []
        for values in period_arrays:
            count = len(values)
            if count == 0:
                continue
            starts = rng.integers(0, count, size=int(np.ceil(count / block_length)))
            rebuilt = [
                values[(start + offset) % count]
                for start in starts
                for offset in range(block_length)
            ]
            sampled.extend(rebuilt[:count])
        draws[draw] = float(np.mean(sampled)) if sampled else np.nan
    return {
        "observed_session_mean_difference_bps": observed,
        "bootstrap_lower_95_bps": float(np.nanquantile(draws, 0.025)),
        "bootstrap_upper_95_bps": float(np.nanquantile(draws, 0.975)),
    }


def acceptance_diagnostics(
    source: pd.DataFrame,
    *,
    round_trip_cost_bps: float,
) -> pd.DataFrame:
    """Attach the fixed zero/cost bins; never use target-derived quantiles."""

    frame = source.copy()
    balance = pd.to_numeric(frame["acceptance_balance_bps"], errors="coerce")
    cost = float(round_trip_cost_bps)
    frame["acceptance_bin"] = np.select(
        [balance.le(-cost), balance.le(0.0), balance.le(cost)],
        [
            "acceptance_balance<=-cost",
            "-cost<acceptance_balance<=0",
            "0<acceptance_balance<=cost",
        ],
        default="acceptance_balance>cost",
    )
    return frame


def acceptance_spearman(source: pd.DataFrame) -> tuple[float, float]:
    """Continuous diagnostic with no searched cutoff."""

    frame = source[["acceptance_balance_bps", "net_payoff_bps"]].dropna()
    if len(frame) < 3:
        return (float("nan"), float("nan"))
    statistic = spearmanr(frame["acceptance_balance_bps"], frame["net_payoff_bps"])
    return (float(statistic.statistic), float(statistic.pvalue))
